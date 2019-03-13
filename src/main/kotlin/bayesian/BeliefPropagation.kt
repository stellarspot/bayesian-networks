package bayesian

import bayesian.core.BayesianNetwork
import bayesian.core.Evidence

data class VariableNode(val name: String, var domainSize: Int = -1, val factors: MutableList<FactorNode> = mutableListOf()) {
    override fun toString() = "$name[$domainSize]"
}

data class FactorNode(val variables: MutableList<VariableNode> = mutableListOf()) {
    override fun toString() = variables.map { it.name }.joinToString("-")
}

class FactorGraph(bayesianNetwork: BayesianNetwork) {
    val factors = mutableListOf<FactorNode>()
    val variablesMap = mutableMapOf<String, VariableNode>()

    init {

        for (node in bayesianNetwork.nodes) {
            val variable = VariableNode(node.name, node.domain.size)
            variablesMap[node.name] = variable

            val factor = FactorNode()
            factor.variables.add(variable)
            for (parent in node.parents) {
                val v = variablesMap.getOrPut(parent.name) { VariableNode(parent.name) }
                factor.variables.add(v)
            }
            factors.add(factor)
        }
    }

    fun dump() {
        for (variable in variablesMap.values) {
            println("variable: $variable")
        }

        for (factor in factors) {
            println("factor: $factor")
        }
    }
}

fun BayesianNetwork.beliefPropagation(vararg evidences: Evidence): Double {
    val graph = FactorGraph(this)
    graph.dump()
    return 0.0
}