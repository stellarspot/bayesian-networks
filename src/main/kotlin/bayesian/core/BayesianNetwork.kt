package bayesian.core

import java.lang.Exception

interface ProbabilityTable {
    operator fun get(key: List<String>): Double
}

class MapProbabilityTable(val map: Map<List<String>, Double>) : ProbabilityTable {
    override fun get(key: List<String>) = map.getOrElse(key) { throw Exception("Unknown key: ${key.joinToString()}") }
}

data class Evidence(val name: String, val value: String)

class Node(val name: String,
           val domain: List<String>,
           val probabilityTable: ProbabilityTable,
           vararg var parents: Node)

class BayesianNetwork(vararg val nodes: Node) {
    private val map = mutableMapOf<String, Node>()

    init {
        for (node in nodes) {
            map[node.name] = node
        }
    }

    operator fun get(name: String): Node = map.getOrElse(name) {
        throw Exception("Bayesian network does not contain node: $name")
    }
}