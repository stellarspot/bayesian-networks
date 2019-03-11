package bayesian.core

import java.lang.Exception

interface ProbabilityTable {
    operator fun get(key: List<String>): Double
}

class MapProbabilityTable(val map: Map<List<String>, Double>) : ProbabilityTable {
    override fun get(key: List<String>) = map.getOrElse(key) { throw Exception("Unknown key: ${key.joinToString()}") }
}

class Node(val name: String,
           val domain: List<String>,
           val probabilityTable: ProbabilityTable,
           vararg val parents: Node)

class BayesianNetwork(vararg val nodes: Node)