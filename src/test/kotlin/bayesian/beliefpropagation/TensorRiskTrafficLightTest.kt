package bayesian.beliefpropagation

import bayesian.core.*
import bayesian.util.assertDoubleEquals
import org.junit.Assert
import org.junit.Test

class TensorRiskTrafficLightTest {

    private val trafficLightProbability = MapProbabilityTable(
            mapOf(
                    listOf("green") to 0.4,
                    listOf("yellow") to 0.25,
                    listOf("red") to 0.35))

    private val riskProbability = MapProbabilityTable(
            mapOf(
                    listOf("green", "high") to 0.1,
                    listOf("green", "low") to 0.9,
                    listOf("yellow", "high") to 0.55,
                    listOf("yellow", "low") to 0.45,
                    listOf("red", "high") to 0.95,
                    listOf("red", "low") to 0.05))

    private val trafficLight = Node("TrafficLight", listOf("green", "yellow", "red"), trafficLightProbability)
    private val risk = Node("Risk", listOf("high", "low"), riskProbability, trafficLight)

    private val bayesianNetwork = BayesianNetwork(trafficLight, risk)

    @Test
    fun testMarginalization() {

        val marginalizationDividend = bayesianNetwork.marginalize(
                Evidence(trafficLight.name, "yellow"),
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.1375, marginalizationDividend)

        val marginalizationDivisor = bayesianNetwork.marginalize(
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.51, marginalizationDivisor)
    }

    @Test
    fun testBeliefPropagation() {


        val marginalizationDividend = bayesianNetwork.beliefPropagation(
                Evidence(trafficLight.name, "yellow"),
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.1375, marginalizationDividend)

        val marginalizationDivisor = bayesianNetwork.beliefPropagation(
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.51, marginalizationDivisor)


        val probabilityTrafficLihghtGivenRisk = marginalizationDividend / marginalizationDivisor

        assertDoubleEquals(0.27, probabilityTrafficLihghtGivenRisk)
    }

    @Test
    fun testBeliefLoopyPropagation() {


        val marginalizationDividend = bayesianNetwork.loopyBeliefPropagation(
                Evidence(trafficLight.name, "yellow"),
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.1375, marginalizationDividend)

        val marginalizationDivisor = bayesianNetwork.loopyBeliefPropagation(
                Evidence(risk.name, "high"))

        assertDoubleEquals(0.51, marginalizationDivisor)


        val probabilityTrafficLihghtGivenRisk = marginalizationDividend / marginalizationDivisor

        assertDoubleEquals(0.27, probabilityTrafficLihghtGivenRisk)
    }
}