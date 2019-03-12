package bayesian.sample

import bayesian.core.BayesianNetwork
import bayesian.core.MapProbabilityTable
import bayesian.core.Node
import bayesian.draw.openrndr.draw
import bayesian.parser.BayesianNetworkBifParser
import bayesian.parser.ProbabilityItem
import bayesian.parser.VariableItem

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Provide path to Bayesian Network file with *.bif extension")
        System.exit(1)
    }
    val file = args[0]

    println("file path: $file")

    data class NodeItem(val prob: ProbabilityItem, val node: Node)

    val variablesMap = mutableMapOf<String, VariableItem>()
    val nodesMap = mutableMapOf<String, NodeItem>()

    val nodesList = mutableListOf<Node>()
    val parser = BayesianNetworkBifParser(file)

    parser.parse {

        if (it is VariableItem) {
            variablesMap[it.name] = it
        } else if (it is ProbabilityItem) {
            val name = it.name
            val domain = variablesMap.getOrElse(name) {
                throw Exception("Variable is absent: $name")
            }.domain

            val node = Node(it.name,
                    domain,
                    MapProbabilityTable(mapOf()))

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

    val network = BayesianNetwork(*nodesList.toTypedArray())
    draw(network, "Parsed Bayesian Network", 1000, 600, 10)
}