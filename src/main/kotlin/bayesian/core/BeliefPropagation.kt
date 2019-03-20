package bayesian.core

import bayesian.util.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory


fun BayesianNetwork.beliefPropagation(vararg evidences: Evidence): Double {

    if (evidences.isEmpty()) {
        return 1.0
    }

    val graph = FactorGraph(this)
    graph.applyEvidences(*evidences)
    // graph.dump()
    graph.sendMessages()
    return graph.calculateMarginalization(*evidences)
}

fun BayesianNetwork.loopyBeliefPropagation(vararg evidences: Evidence, maxSteps: Int = 100): Double {

    if (evidences.isEmpty()) {
        return 1.0
    }

    val graph = FactorGraph(this)
    graph.applyEvidences(*evidences)
    graph.dump()
    graph.sendLoopyMessages(maxSteps)
    return graph.calculateMarginalization(*evidences)
}

data class VariableNode(val name: String,
                        var domainSize: Int = -1,
                        val inEdges: MutableList<Edge> = mutableListOf(),
                        val outEdges: MutableList<Edge> = mutableListOf(),
                        var evidenceIndex: Int = -1) {

    fun addEdge(factor: FactorNode) {
        val edge = Edge(factor, this)
        outEdges.add(edge)
        edge.factor.inEdges.add(edge)
    }

    override fun toString() = "Variable-$name"
}

data class FactorNode(var tensor: INDArray,
                      val inEdges: MutableList<Edge> = mutableListOf(),
                      val outEdges: MutableList<Edge> = mutableListOf()) {

    fun addEdge(variable: VariableNode) {
        val edge = Edge(this, variable)
        outEdges.add(edge)
        edge.variable.inEdges.add(edge)
    }


    override fun toString() = "Factor-" + outEdges.map { it.variable.name }.joinToString("-")
}

data class Edge(val factor: FactorNode, val variable: VariableNode, var message: INDArray? = null)

private fun getTensor(node: Node): INDArray {

    val probability = node.probabilityTable

    val vector = mutableListOf<Double>()
    val domains = mutableListOf<List<String>>()
    for (domain in node.parents.map { it.domain }) {
        domains.add(domain)
    }
    domains.add(node.domain)

    val arguments = mutableListOf<String>()

    fun iterateValues(index: Int) {
        if (index == domains.size) {
            val p = probability[arguments]
            vector.add(p)
        } else {
            for (arg in domains[index]) {
                arguments.add(arg)
                iterateValues(index + 1)
                arguments.removeAt(index)
            }
        }
    }

    iterateValues(0)
    return Nd4j.create(vector.toDoubleArray(), domains.map { it.size }.toIntArray())
}

class FactorGraph(val bayesianNetwork: BayesianNetwork) {

    private val logger = LoggerFactory.getLogger(javaClass)

    private val factors = mutableListOf<FactorNode>()
    private val variablesMap = mutableMapOf<String, VariableNode>()

    init {

        fun getVariable(name: String) =
                variablesMap.getOrPut(name) { VariableNode(name) }

        for (node in bayesianNetwork.nodes) {
            val variable = getVariable(node.name)
            variable.domainSize = node.domain.size
            variablesMap[node.name] = variable

            val factor = FactorNode(getTensor(node))
            variable.addEdge(factor)
            for (parent in node.parents) {
                val v = getVariable(parent.name)
                factor.addEdge(v)
                v.addEdge(factor)
            }
            factor.addEdge(variable)
            factors.add(factor)
        }
    }

    fun applyEvidences(vararg evidences: Evidence) {
        for (evidence in evidences) {
            val name = evidence.name
            val variable = variablesMap[name]!!
            variable.domainSize = 1
            val node = bayesianNetwork[name]!!

            val evidenceIndex = node.domain.indexOf(evidence.value)
            if (evidenceIndex < 0) {
                throw Exception("Node $name does not contain value ${evidence.value}")
            }
            variable.evidenceIndex = evidenceIndex
        }

        for (factor in factors) {
            var tensor = factor.tensor
            val variables = factor.outEdges.map { it.variable }
            for (index in (variables.size - 1 downTo 0)) {
                val variable = variables[index]
                val evidenceIndex = variable.evidenceIndex
                if (evidenceIndex != -1) {
                    tensor = takeTensor(tensor, index, evidenceIndex)
                }
            }
            factor.tensor = tensor
        }
    }

    fun sendMessages() {

        val variables = variablesMap.values
        var step = 0
        do {

            var finished = true
            logger.debug("step: ${step++}")

            for (variable in variables) {
                for (edge in variable.outEdges) {
                    if (edge.message == null) {
                        finished = false
                        sendMessage(variable, edge)
                    }
                }
            }

            for (factor in factors) {
                for ((index, edge) in factor.outEdges.withIndex()) {
                    if (edge.message == null) {
                        finished = false
                        sendMessage(factor, edge, index)
                    }
                }
            }

        } while (!finished)
    }

    fun sendLoopyMessages(maxSteps: Int) {

        val variables = variablesMap.values

        for (variable in variables) {
            for (edge in variable.outEdges) {
                edge.message = initialMessage(variable.domainSize)
            }
        }


        var step = 0
        do {

            logger.debug("step: ${step++}")

            for (variable in variables) {
                for (edge in variable.outEdges) {
                    sendMessage(variable, edge)
                }
            }

            for (factor in factors) {
                for ((index, edge) in factor.outEdges.withIndex()) {
                    sendMessage(factor, edge, index)
                }
            }

        } while (step < maxSteps)
    }

    private fun sendMessage(variable: VariableNode, edge: Edge) {
        val edges = variable.outEdges
        if (edges.size == 1) {
            edge.message = initialMessage(variable.domainSize)
            logger.debug("send message(v->f): $variable, ${edge.factor}, initial ${edge.message}")
        } else {

            val inEdges = variable.inEdges.filter { it.factor != edge.factor }
            val canSendMessage = inEdges.all { it.message != null }

            if (canSendMessage) {
                val tensors = inEdges
                        .map { it.message }
                        .filterNotNull()
                assert(tensors.size == inEdges.size)

                var tensor = initialMessage(variable.domainSize)
                for (t in tensors) {
                    tensor = multiplyMessageElementWise(tensor, t)
                }
                edge.message = tensor
                logger.debug("send message(v->f): $variable, ${edge.factor}:\n${edge.message}")
            }
        }
    }

    private fun sendMessage(factor: FactorNode, edge: Edge, index: Int) {

        val edges = factor.outEdges
        if (edges.size == 1) {
            edge.message = factor.tensor
            logger.debug("send message(f->v): $factor, ${edge.variable}, initial: ${edge.message}")
        } else {

            // in edges must have the same order as outedges
            // they correspond to arguments order in the probability tensor
            val inEdges = edges
                    .map {
                        it.variable.outEdges.find { e -> e.factor == factor }
                    }.filterNotNull()

            assert(inEdges.size == edges.size)

            val canSendMessage = inEdges
                    .filterIndexed { i, _ -> i != index }
                    .all { it.message != null }

            if (canSendMessage) {

                val messages = inEdges.map { it.message }.toTypedArray()
                edge.message = multiplyTensorMessages(factor.tensor, index, *messages)
                logger.debug("send message(f->v): $factor, ${edge.variable}:\n${edge.message}")
            }

        }
    }


    fun calculateMarginalization(vararg evidences: Evidence): Double {

        val evidence = evidences[0]

        val variable = variablesMap.getOrElse(evidence.name) {
            throw Exception("There is no node for evidence: ${evidence.name}")
        }

        val edge = variable.outEdges[0]
        val messageOut = edge.message!!
        val messageIn = edge.factor.outEdges.find { it.variable.name == variable.name }!!.message!!

        val result = multiplyTensorMessage(messageIn, messageOut).getDouble(0)

        return result
    }

    fun dump() {
        for (variable in variablesMap.values) {
            println("variable: $variable")
        }

        for (factor in factors) {
            println("factor: $factor, tensor:\n${factor.tensor}")
        }
    }
}