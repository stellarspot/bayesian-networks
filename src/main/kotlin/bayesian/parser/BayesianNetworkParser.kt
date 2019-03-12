package bayesian.parser


interface ParserItem
data class NetworkItem(val name: String) : ParserItem

data class VariableItem(val name: String, val domain: List<String>) : ParserItem
data class ProbabilityItem(val name: String, val parents: List<String>) : ParserItem

interface BayesianNetworkParser {

    fun parse(consumer: (ParserItem) -> Unit)

}