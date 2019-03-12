package bayesian.parser

import java.io.BufferedReader
import java.io.File

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


class BayesianNetworkBifParser(val file: String) : BayesianNetworkParser {

    override fun parse(consumer: (ParserItem) -> Unit) {
        File(file)
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
            domain = domainString.split(", ")
        }

        if (domain == null) {
            throw Exception("Domain is not found for variable: $name")
        }

        return VariableItem(name, domain)
    }

    private fun parseProbability(line: String, reader: BufferedReader): ParserItem {

        while (true) {
            val nextLine = reader.readLine() ?: throw Exception("Unexpected end of file!")

            if (PATTERN_END.matches(nextLine)) {
                break
            }
        }
        return ProbabilityItem("UNKNOWN")
    }
}