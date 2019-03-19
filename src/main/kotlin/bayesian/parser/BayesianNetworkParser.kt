package bayesian.parser

import bayesian.core.BayesianNetwork
import bayesian.core.MapProbabilityTable
import bayesian.core.Node
import java.io.File
import java.net.URL


interface ParserItem

data class NetworkItem(val name: String) : ParserItem
data class VariableItem(val name: String, val domain: List<String>) : ParserItem
data class ProbabilityItem(val name: String,
                           val parents: List<String>,
                           val probabilities: List<ProbabilityTableItem>) : ParserItem

data class ProbabilityTableItem(val arguments: List<String>, val values: List<Double>)

interface BayesianNetworkParser {

    fun parse(consumer: (ParserItem) -> Unit)

}

fun parse(url: URL): BayesianNetwork {
    data class NodeItem(val prob: ProbabilityItem, val node: Node)

    val variablesMap = mutableMapOf<String, VariableItem>()
    val nodesMap = mutableMapOf<String, NodeItem>()

    val nodesList = mutableListOf<Node>()
    val parser = BayesianNetworkBifParser(url)

    parser.parse {

        if (it is VariableItem) {
            variablesMap[it.name] = it
        } else if (it is ProbabilityItem) {

            val name = it.name
            val domain = variablesMap.getOrElse(name) {
                throw Exception("Variable is absent: $name")
            }.domain

            val probabilityTable = mutableMapOf<List<String>, Double>()
            for (probItem in it.probabilities) {
                val args = probItem.arguments
                val values = probItem.values

                for ((i, domainArg) in domain.withIndex()) {
                    val argsWithDomain = args.toMutableList()
                    argsWithDomain.add(domainArg)
                    probabilityTable[argsWithDomain] = values[i]
                }
            }

            val node = Node(it.name,
                    domain,
                    MapProbabilityTable(probabilityTable))

            nodesMap[name] = NodeItem(it, node)
            nodesList.add(node)
        }
    }

    for (node in nodesList) {
        val parents = mutableListOf<Node>()
        val nodeItem = nodesMap.getOrElse(node.name) {
            throw Exception("Node is absent: ${node.name}")
        }

        for (parentName in nodeItem.prob.parents) {
            val parent = nodesMap.getOrElse(parentName) {
                throw Exception("Parent is absent: ${parentName}")
            }
            parents.add(parent.node)
        }

        node.parents = parents.toTypedArray()
    }

    return BayesianNetwork(*nodesList.toTypedArray())
}