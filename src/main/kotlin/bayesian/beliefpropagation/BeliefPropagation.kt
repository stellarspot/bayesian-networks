package bayesian.beliefpropagation

import bayesian.core.BayesianNetwork
import bayesian.core.Evidence
import bayesian.core.Node
import bayesian.util.takeTensor
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

data class VariableNode(val name: String,
                        var domainSize: Int = -1,
                        val edges: MutableList<VariableFactorEdge> = mutableListOf(),
                        var evidenceIndex: Int = -1) {
    override fun toString() = "$name[$domainSize]"
}

data class FactorNode(var tensor: INDArray,
                      val edges: MutableList<FactorVariableEdge> = mutableListOf()) {
    override fun toString() = edges.map { it.variable.name }.joinToString("-")
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
            for (parent in node.parents) {
                val v = getVariable(parent.name)
                factor.edges.add(FactorVariableEdge(v))
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
    val graph = FactorGraph(this)
    graph.applyEvidences(*evidences)
    // graph.dump()
    return 1.0
}