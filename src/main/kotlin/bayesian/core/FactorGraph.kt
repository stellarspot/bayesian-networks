package bayesian.core

import bayesian.util.multiplyTensorMessage
import bayesian.util.takeTensor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory

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

data class Edge(val factor: FactorNode,
                val variable: VariableNode,
                var message: INDArray? = null,
                var prevMessage: INDArray? = null)

class FactorGraph(val bayesianNetwork: BayesianNetwork) {

    val logger = LoggerFactory.getLogger(javaClass)

    val factors = mutableListOf<FactorNode>()
    val variablesMap = mutableMapOf<String, VariableNode>()

    init {

        fun getVariable(name: String) =
                variablesMap.getOrPut(name) { VariableNode(name) }

        for (node in bayesianNetwork.nodes) {

            val variable = getVariable(node.name)
            variable.domainSize = node.domain.size
            variablesMap[node.name] = variable

            val factor = FactorNode(getTensor(node))
            for (parent in node.parents) {
                val v = getVariable(parent.name)
                factor.addEdge(v)
                v.addEdge(factor)
            }

            factor.addEdge(variable)
            variable.addEdge(factor)

            factors.add(factor)
        }
    }

    fun applyEvidences(vararg evidences: Evidence) {
        for (evidence in evidences) {
            val name = evidence.name
            val variable = variablesMap[name]!!
            variable.domainSize = 1
            val node = bayesianNetwork[name]

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


    fun calculateMarginalization(vararg evidences: Evidence): Double {

        val evidence = evidences.first()

        val variable = variablesMap.getOrElse(evidence.name) {
            throw Exception("There is no node for evidence: ${evidence.name}")
        }

        val messageIn = variable.inEdges.first().message!!
        val messageOut = variable.outEdges.first().message!!

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


fun getTensor(node: Node): INDArray {

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
