package bayesian.beliefpropagation

import bayesian.core.BayesianNetwork
import bayesian.core.Evidence
import bayesian.core.Node
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

data class VariableNode(val name: String, var domainSize: Int = -1, val edges: MutableList<VariableFactorEdge> = mutableListOf()) {
    override fun toString() = "$name[$domainSize]"
}

data class FactorNode(val tensor: INDArray,
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

class FactorGraph(bayesianNetwork: BayesianNetwork) {
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
            factor.edges.add(FactorVariableEdge(variable))
            for (parent in node.parents) {
                val v = getVariable(parent.name)
                factor.edges.add(FactorVariableEdge(v))
            }
            factors.add(factor)
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
    graph.dump()
    return 0.0
}