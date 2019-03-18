package bayesian.sample

import bayesian.beliefpropagation.beliefPropagation
import bayesian.core.BayesianNetwork
import bayesian.core.Evidence
import bayesian.core.MapProbabilityTable
import bayesian.core.Node
import bayesian.draw.openrndr.draw

fun main(args: Array<String>) {

    val trafficLightProbability = MapProbabilityTable(
            mapOf(
                    listOf("green") to 0.4,
                    listOf("yellow") to 0.25,
                    listOf("red") to 0.35)
    )

    val riskProbability = MapProbabilityTable(
            mapOf(
                    listOf("green", "high") to 0.1,
                    listOf("green", "low") to 0.9,
                    listOf("yellow", "high") to 0.55,
                    listOf("yellow", "low") to 0.45,
                    listOf("red", "high") to 0.95,
                    listOf("red", "low") to 0.05
            )
    )

    val trafficLight = Node("TrafficLight", listOf("green", "yellow", "red"), trafficLightProbability)
    val risk = Node("Risk", listOf("high", "low"), riskProbability, trafficLight)
    val bayesianNetwork = BayesianNetwork(trafficLight, risk)

    val marginalizationDivisor = bayesianNetwork.beliefPropagation(
            Evidence(risk.name, "high"))

    val marginalizationDividend = bayesianNetwork.beliefPropagation(
            Evidence(trafficLight.name, "yellow"),
            Evidence(risk.name, "high"))


    val probabilityTrafficLihghtGivenRisk = marginalizationDividend / marginalizationDivisor

    println("Probability Traffic Light is yellow given Risk is High: $probabilityTrafficLihghtGivenRisk")

    draw(bayesianNetwork)
}