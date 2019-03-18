package bayesian.core

import java.lang.Exception

fun BayesianNetwork.marginalize(vararg evidences: Evidence): Double {

    if (evidences.isEmpty()) {
        return 1.0
    }

    data class DomainWithCount(val domain: List<String>, var count: Int = 0) {
        fun value() = domain[count]
    }

    val domainMap = mutableMapOf<String, DomainWithCount>()

    evidences.forEach { domainMap[it.name] = DomainWithCount(listOf(it.value)) }

    val nodes = this.nodes

    nodes.forEach {
        domainMap.putIfAbsent(it.name, DomainWithCount(it.domain))
    }

    fun probability(): Double {
        var p = 1.0
        for (node in nodes) {
            val values = node.parents.map { domainMap[it.name]!!.value() }.toMutableList()
            values.add(domainMap[node.name]!!.value())
            p *= node.probabilityTable[values]

        }
        return p
    }

    var p = 0.0

    while (true) {

        p += probability()

        var index = 0
        while (true) {

            if (index == nodes.size) {
                return p
            }

            val d = domainMap[nodes[index].name]!!
            d.count++
            if (d.count == d.domain.size) {
                d.count = 0
                index++
            } else {
                break
            }
        }
    }

    throw Exception("Unexpected end of the marginalization algorithm!")
}