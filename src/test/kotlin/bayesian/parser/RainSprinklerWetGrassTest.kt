package bayesian.parser

import bayesian.core.Evidence
import bayesian.core.loopyBeliefPropagation
import bayesian.core.marginalize
import bayesian.util.assertDoubleEquals
import bayesian.util.getResourceUrl
import org.junit.Assert
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class RainSprinklerWetGrassTest {

    private val file = "parser/rain_sprinkler_wet_grass.biff"

    @Test
    fun testGraphParsing() {

        val network = getNetwork()
        val nodes = network.nodes

        assertEquals(3, nodes.size)

        val rain = nodes[0]
        assertEquals("rain", rain.name)
        assertEquals(listOf("true", "false"), rain.domain)
        assertTrue(rain.parents.isEmpty())

        assertDoubleEquals(0.2, rain.probabilityTable[listOf("true")])
        assertDoubleEquals(0.8, rain.probabilityTable[listOf("false")])

        val sprinkler = nodes[1]
        assertEquals("sprinkler", sprinkler.name)
        assertEquals(listOf("switch_on", "switch_off"), sprinkler.domain)
        Assert.assertArrayEquals(arrayOf(rain), sprinkler.parents)

        assertDoubleEquals(0.01, sprinkler.probabilityTable[listOf("true", "switch_on")])
        assertDoubleEquals(0.99, sprinkler.probabilityTable[listOf("true", "switch_off")])
        assertDoubleEquals(0.4, sprinkler.probabilityTable[listOf("false", "switch_on")])
        assertDoubleEquals(0.6, sprinkler.probabilityTable[listOf("false", "switch_off")])

        val wetGrass = nodes[2]
        assertEquals("wet_grass", wetGrass.name)
        assertEquals(listOf("wet", "dry"), wetGrass.domain)
        Assert.assertArrayEquals(arrayOf(sprinkler, rain), wetGrass.parents)

        assertDoubleEquals(0.99, wetGrass.probabilityTable[listOf("switch_on", "true", "wet")])
        assertDoubleEquals(0.01, wetGrass.probabilityTable[listOf("switch_on", "true", "dry")])
        assertDoubleEquals(0.9, wetGrass.probabilityTable[listOf("switch_on", "false", "wet")])
        assertDoubleEquals(0.1, wetGrass.probabilityTable[listOf("switch_on", "false", "dry")])
        assertDoubleEquals(0.8, wetGrass.probabilityTable[listOf("switch_off", "true", "wet")])
        assertDoubleEquals(0.2, wetGrass.probabilityTable[listOf("switch_off", "true", "dry")])
        assertDoubleEquals(0.0, wetGrass.probabilityTable[listOf("switch_off", "false", "wet")])
        assertDoubleEquals(1.0, wetGrass.probabilityTable[listOf("switch_off", "false", "dry")])
    }

    @Test
    fun testMarginalization() {

        val network = getNetwork()

        val rain = network.nodes[0]
        val sprinkler = network.nodes[1]
        val wetGrass = network.nodes[2]

        // P(WG=wet)
        val marginalizationWG = network.marginalize(
                Evidence(wetGrass.name, "wet"))

        assertDoubleEquals(0.4483, marginalizationWG)

        // P(R=true, WG=wet)
        val marginalizationRWG = network.marginalize(
                Evidence(rain.name, "true"),
                Evidence(wetGrass.name, "wet"))


        assertDoubleEquals(0.1603, marginalizationRWG)

        // P(R=true, Sprinkler=switch_on, WG=wet)
        val marginalizationRSWG = network.marginalize(
                Evidence(rain.name, "true"),
                Evidence(sprinkler.name, "switch_on"),
                Evidence(wetGrass.name, "wet"))

        assertDoubleEquals(0.00198, marginalizationRSWG)


        val marginalization = network.marginalize()

        assertDoubleEquals(1.0, marginalization)
    }

    @Test
    fun testLoopyBeliefPropagation() {

        val network = getNetwork()

        val rain = network.nodes[0]
        val sprinkler = network.nodes[1]
        val wetGrass = network.nodes[2]

        // P(R=true, Sprinkler=switch_on, WG=wet)

        val marginalizationRWG = network.loopyBeliefPropagation(
                maxSteps = 20,
                evidences = *arrayOf(
                        Evidence(rain.name, "true"),
                        Evidence(wetGrass.name, "wet")))

        println("marginalizationRWG = $marginalizationRWG")

        // P(R=true, Sprinkler=switch_on, WG=wet)

        val marginalizationRSWG = network.loopyBeliefPropagation(
                maxSteps = 20,
                evidences = *arrayOf(
                        Evidence(rain.name, "true"),
                        Evidence(sprinkler.name, "switch_on"),
                        Evidence(wetGrass.name, "wet")))

        println("marginalizationRSWG = $marginalizationRSWG")
    }

    private fun getNetwork() = parse(getResourceUrl(file)!!)
}
