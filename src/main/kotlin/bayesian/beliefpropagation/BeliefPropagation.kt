package bayesian.beliefpropagation

import bayesian.core.BayesianNetwork
import bayesian.core.Evidence
import org.nd4j.linalg.api.ndarray.INDArray

data class VariableNode(val name: String, var domainSize: Int = -1, val edges: MutableList<VariableFactorEdge> = mutableListOf()) {
    override fun toString() = "$name[$domainSize]"
}

data class FactorNode(val edges: MutableList<FactorVariableEdge> = mutableListOf()) {
    override fun toString() = edges.map { it.variable.name }.joinToString("-")
}

data class VariableFactorEdge(val factor: FactorNode, var message: INDArray? = null)
data class FactorVariableEdge(val variable: VariableNode, var message: INDArray? = null)

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

            val factor = FactorNode()
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
            println("factor: $factor")
        }
    }
}

fun BayesianNetwork.beliefPropagation(vararg evidences: Evidence): Double {
    val graph = FactorGraph(this)
    graph.dump()
    return 0.0
}