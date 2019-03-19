package bayesian.parser


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