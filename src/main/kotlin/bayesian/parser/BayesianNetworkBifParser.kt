package bayesian.parser

import java.io.BufferedReader
import java.io.File
import java.net.URL

enum class ElementType {
    NETWORK,
    VARIABLE,
    PROBABILITY,
    UNKNOWN,
}

private val PATTERN_END = """^\}""".toRegex()
private val PATTERN_NETWORK = """^network (\w+) \{""".toRegex()
private val PATTERN_VARIABLE = """^variable (\w+) \{""".toRegex()
private val PATTERN_VARIABLE_TYPE = """\s+type (\w+) \[ (\d+) \] \{ ((\w+,?\s)+)\};""".toRegex()
private val PATTERN_PROBABILITY = """^probability \( (\w+)( \| \w+(, \w+)*)? \) \{""".toRegex()

private val PATTERN_NUMBER = """\d+[\.]\d+"""
private val PATTERN_PROBABILITY_TABLE = """\s+table (($PATTERN_NUMBER)(,\s$PATTERN_NUMBER)*);""".toRegex()
private val PATTERN_PROBABILITY_ARGUMENTS_ARGS = """\s+\(((\w+)(,\s\w+)*)\).*;""".toRegex()
private val PATTERN_PROBABILITY_ARGUMENTS_VALS = """\s+\(.*\) (($PATTERN_NUMBER)(,\s$PATTERN_NUMBER)*);""".toRegex()

class BayesianNetworkBifParser(val url: URL) : BayesianNetworkParser {

    override fun parse(consumer: (ParserItem) -> Unit) {

        url
                .openStream()
                .bufferedReader(Charsets.UTF_8)
                .use {
                    while (true) {
                        val line = it.readLine() ?: break
                        val type = getType(line)
                        val parseItem = parse(type, line, it)
                        consumer(parseItem)
                    }
                }
    }

    private fun getType(line: String): ElementType = when {
        line.startsWith("network") -> ElementType.NETWORK
        line.startsWith("variable") -> ElementType.VARIABLE
        line.startsWith("probability") -> ElementType.PROBABILITY
        else -> ElementType.UNKNOWN
    }

    private fun parse(type: ElementType, line: String, reader: BufferedReader): ParserItem = when (type) {
        ElementType.NETWORK -> parseNetwork(line, reader)
        ElementType.VARIABLE -> parseVariable(line, reader)
        ElementType.PROBABILITY -> parseProbability(line, reader)
        else -> throw Exception("Unknown line type: '$line'")
    }

    private fun parseNetwork(line: String, reader: BufferedReader): ParserItem {

        val matchResult = PATTERN_NETWORK.matchEntire(line)
                ?: throw Exception("Unable to parse network type: '$line'")

        val name = matchResult.groupValues[1]

        while (true) {
            val nextLine = reader.readLine() ?: throw Exception("Unexpected end of file!")
            if (PATTERN_END.matches(nextLine)) {
                break
            }
            throw Exception("Unknown lines in network body: '$nextLine'")
        }

        return NetworkItem(name)
    }

    private fun parseVariable(line: String, reader: BufferedReader): ParserItem {

        val matchResult = PATTERN_VARIABLE.matchEntire(line)
                ?: throw Exception("Unable to parse variable type: '$line'")

        val name = matchResult.groupValues[1]

        var domain: List<String>? = null

        while (true) {
            val nextLine = reader.readLine() ?: throw Exception("Unexpected end of file!")
            if (PATTERN_END.matches(nextLine)) {
                break
            }

            val matchTypeResult = PATTERN_VARIABLE_TYPE.matchEntire(nextLine)
                    ?: throw Exception("Unable to parse variable type: '$nextLine'")

            val domainString = matchTypeResult.groupValues[3]
            domain = domainString.split(", ").map { it.trim() }
        }

        if (domain == null) {
            throw Exception("Domain is not found for variable: $name")
        }

        return VariableItem(name, domain)
    }

    private fun parseProbability(line: String, reader: BufferedReader): ParserItem {

        val matchResult = PATTERN_PROBABILITY.matchEntire(line)
                ?: throw Exception("Unable to parse probability type: '$line'")

        val name = matchResult.groupValues[1]

        val parentsString = matchResult.groupValues[2]
        val parents = parentsString.split(" ", "|", ",").filter { it.isNotEmpty() }

        val probabilities = mutableListOf<ProbabilityTableItem>()

        while (true) {
            val nextLine = reader.readLine() ?: throw Exception("Unexpected end of file!")

            if (PATTERN_END.matches(nextLine)) {
                break
            }

            val tableMatch = PATTERN_PROBABILITY_TABLE.matchEntire(nextLine)

            if (tableMatch != null) {
                val values = tableMatch
                        .groupValues[1]
                        .split(", ")
                        .map { it.toDouble() }
                probabilities.add(ProbabilityTableItem(emptyList(), values))
                continue
            }

            val argsMatch = PATTERN_PROBABILITY_ARGUMENTS_ARGS.matchEntire(nextLine)

            if (argsMatch != null) {

                val args = argsMatch.groupValues[1].split(", ")

                val valsMatch = PATTERN_PROBABILITY_ARGUMENTS_VALS.matchEntire(nextLine)!!
                val vals = valsMatch
                        .groupValues[1]
                        .split(", ")
                        .map { it.toDouble() }

                probabilities.add(ProbabilityTableItem(args, vals))
            }
        }

        return ProbabilityItem(name, parents, probabilities)
    }
}