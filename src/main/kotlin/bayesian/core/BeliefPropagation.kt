package bayesian.core

import bayesian.util.initialMessage
import bayesian.util.multiplyMessage
import bayesian.util.takeTensor
import bayesian.util.transposeLastAxisToFirst
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

data class VariableNode(val name: String,
                        var domainSize: Int = -1,
                        val edges: MutableList<VariableFactorEdge> = mutableListOf(),
                        var evidenceIndex: Int = -1) {
    override fun toString() = "Variable-$name"
}

data class FactorNode(var tensor: INDArray,
                      val edges: MutableList<FactorVariableEdge> = mutableListOf()) {
    override fun toString() = "Factor-" + edges.map { it.variable.name }.joinToString("-")
}

data class VariableFactorEdge(val factor: FactorNode, var message: INDArray? = null)
data class FactorVariableEdge(val variable: VariableNode, var message: INDArray? = null)

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
            variable.edges.add(VariableFactorEdge(factor))
            for (parent in node.parents) {
                val v = getVariable(parent.name)
                factor.edges.add(FactorVariableEdge(v))
                v.edges.add(VariableFactorEdge(factor))
            }
            factor.edges.add(FactorVariableEdge(variable))
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
            val variables = factor.edges.map { it.variable }
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
        var finished = false

        val variables = variablesMap.values
        var step = 0
        while (!finished && step++ < 2) {

            logger.debug("step: $step")

            for (variable in variables) {
                for (edge in variable.edges) {
                    if (edge.message == null) {
                        finished = false
                        sendMessage(variable, edge)
                    }
                }
            }
            for (factor in factors) {
                for ((index, edge) in factor.edges.withIndex()) {
                    if (edge.message == null) {
                        finished = false
                        sendMessage(factor, edge, index)
                    }
                }
            }
        }
    }

    private fun sendMessage(variable: VariableNode, edge: VariableFactorEdge) {
        val edges = variable.edges
        if (edges.size == 1) {
            edge.message = initialMessage(variable.domainSize)
            logger.debug("send message(v->f): $variable, ${edge.factor}, initial ${edge.message}")
        } else {

            val inEdges = edges
                    .filter { it != edge }
                    .map {
                        it.factor.edges.find { e -> e.variable == variable }
                    }.filterNotNull()
//            assert(inEdges.size == edges.size)

            val canSendMessage = inEdges.all { it.message != null }
            if (canSendMessage) {
                val tensors = inEdges
                        .map { it.message }
                        .filterNotNull()
                assert(tensors.size == inEdges.size)

                var tensor = initialMessage(variable.domainSize)
                for (t in tensors) {
                    tensor = tensor.mul(t)
                }
                edge.message = tensor
                logger.debug("send message(v->f): $variable, ${edge.factor}:\n${edge.message}")
            }
        }
    }

    private fun sendMessage(factor: FactorNode, edge: FactorVariableEdge, index: Int) {

        val edges = factor.edges
        if (edges.size == 1) {
            edge.message = factor.tensor
            logger.debug("send message(f->v): $factor, ${edge.variable}, initial: ${edge.message}")
        } else {
            val inEdges = edges
                    .map {
                        it.variable.edges.find { e -> e.factor == factor }
                    }.filterNotNull()

            assert(inEdges.size == edges.size)

            val canSendMessage = inEdges
                    .filterIndexed { i, _ -> i != index }
                    .all { it.message != null }

            if (canSendMessage) {
                var tensor = factor.tensor

                for (i in (edges.size - 1 downTo 0)) {
                    val e = inEdges[i]
                    if (i == index) {
                        if (i != 0) {
                            tensor = transposeLastAxisToFirst(tensor)
                        }
                    } else {
                        val message = e.message!!
                        tensor = multiplyMessage(tensor, message)
                    }
                }
                edge.message = tensor
                logger.debug("send message(f->v): $factor, ${edge.variable}:\n${tensor}")
            }

        }
    }


    fun calculateMarginalization(vararg evidences: Evidence): Double {

        val evidence = evidences[0]

        val variable = variablesMap.getOrElse(evidence.name) {
            throw Exception("There is no node for evidence: ${evidence.name}")
        }

        val edge = variable.edges[0]
        val messageOut = edge.message!!
        val messageIn = edge.factor.edges.find { it.variable.name == variable.name }!!.message!!

        val result = multiplyMessage(messageIn, messageOut).getDouble(0)

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