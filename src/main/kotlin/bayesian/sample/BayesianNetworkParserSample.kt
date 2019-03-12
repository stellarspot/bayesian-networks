package bayesian.sample

import bayesian.core.BayesianNetwork
import bayesian.core.MapProbabilityTable
import bayesian.core.Node
import bayesian.draw.openrndr.draw
import bayesian.parser.BayesianNetworkBifParser
import bayesian.parser.VariableItem

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Provide path to Bayesian Network file with *.bif extension")
        System.exit(1)
    }
    val file = args[0]

    println("file path: $file")

    val nodesMap = mutableMapOf<String, Node>()
    val nodesList = mutableListOf<Node>()
    val parser = BayesianNetworkBifParser(file)
    parser.parse {
        if (it is VariableItem) {

            val node = Node(it.name, it.domain, MapProbabilityTable(mapOf()))
            nodesList.add(node)
        }
    }

    val network = BayesianNetwork(*nodesList.toTypedArray())
    draw(network, "Parsed Bayesian Network", 1000, 600, 20)
}