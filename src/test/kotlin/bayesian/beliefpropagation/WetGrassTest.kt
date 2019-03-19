package bayesian.beliefpropagation

import bayesian.core.*
import bayesian.util.assertDoubleEquals
import org.junit.Assert
import org.junit.Test

class WetGrassTest {


    val rainProbability = MapProbabilityTable(
            mapOf(
                    listOf("true") to 0.2,
                    listOf("false") to 0.8)
    )

    val sprinklerProbability = MapProbabilityTable(
            mapOf(
                    listOf("switch-on") to 0.1,
                    listOf("switch-off") to 0.9)
    )

    val watsonGrassProbability = MapProbabilityTable(
            mapOf(
                    listOf("true", "wet") to 1.0,
                    listOf("true", "dry") to 0.0,
                    listOf("false", "wet") to 0.2,
                    listOf("false", "dry") to 0.8
            )
    )

    val holmesGrassProbability = MapProbabilityTable(
            mapOf(
                    listOf("switch-on", "true", "wet") to 1.0,
                    listOf("switch-on", "true", "dry") to 0.0,
                    listOf("switch-on", "false", "wet") to 0.9,
                    listOf("switch-on", "false", "dry") to 0.1,

                    listOf("switch-off", "true", "wet") to 1.0,
                    listOf("switch-off", "true", "dry") to 0.0,
                    listOf("switch-off", "false", "wet") to 0.0,
                    listOf("switch-off", "false", "dry") to 1.0
            )
    )

    val rain = Node("Rain", listOf("true", "false"), rainProbability)
    val sprinkler = Node("Sprinkler", listOf("switch-on", "switch-off"), sprinklerProbability)
    val watsonGrass = Node("WatsonGrass", listOf("wet", "dry"), watsonGrassProbability, rain)
    val holmesGrass = Node("HolmesGrass", listOf("wet", "dry"), holmesGrassProbability, sprinkler, rain)
    val bayesianNetwork = BayesianNetwork(rain, sprinkler, watsonGrass, holmesGrass)


    @Test
    fun testMarginalization() {

        val marginalizationDivisor = bayesianNetwork.marginalize(
                Evidence(holmesGrass.name, "wet"),
                Evidence(rain.name, "true"))

        assertDoubleEquals(0.2, marginalizationDivisor)

        val marginalizationDividend = bayesianNetwork.marginalize(
                Evidence(holmesGrass.name, "wet"))

        assertDoubleEquals(0.272, marginalizationDividend)

    }

    @Test
    fun testBeliefPropagation() {

        val marginalizationDivisor = bayesianNetwork.beliefPropagation(
                Evidence(holmesGrass.name, "wet"),
                Evidence(rain.name, "true"))

        assertDoubleEquals(0.2, marginalizationDivisor)

        val marginalizationDividend = bayesianNetwork.beliefPropagation(
                Evidence(holmesGrass.name, "wet"))

        assertDoubleEquals(0.272, marginalizationDividend)
    }
}