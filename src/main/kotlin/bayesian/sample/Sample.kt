package bayesian.sample

import bayesian.core.BayesianNetwork
import bayesian.core.MapProbabilityTable
import bayesian.core.Node
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.Drawer
import org.openrndr.draw.FontImageMap
import org.openrndr.draw.FontMap
import org.openrndr.math.Vector2
import java.lang.Exception

fun draw(drawer: Drawer, font: FontMap, network: BayesianNetwork) {

    drawer.fontMap = font


    drawer.background(ColorRGBa.WHITE)
    val center = drawer.bounds.center

    val R = 0.4 * Math.min(drawer.bounds.width, drawer.bounds.height)
    val r = 0.2 * R
    val rr = 0.03 * R

    val nodes = network.nodes
    val N = nodes.size

    val angle = -Math.PI / 2
    val deltaAngle = 2 * Math.PI / N

    drawer.translate(center)

    fun position(i: Int) =
            Vector2(R * Math.cos(angle + i * deltaAngle),
                    R * Math.sin(angle + i * deltaAngle))

    fun drawArrow(i: Int, j: Int) {

        val p1 = position(i)
        val p2 = position(j)
        val n1 = (p2 - p1).normalized

        val v1 = p1 + n1 * r
        val v2 = p2 - n1 * r

        drawer.fill = ColorRGBa.BLUE
        drawer.stroke = ColorRGBa.BLUE
        drawer.lineSegment(v1, v2)
        drawer.circle(v2, rr)
    }

    val indicesMap = mutableMapOf<Node, Int>()

    for (i in 0 until N) {
        indicesMap[nodes[i]] = i
    }


    for (i in 0 until N) {

        val node = nodes[i]
        val position = position(i)

        drawer.fill = ColorRGBa.fromHex(0xffefdb)
        drawer.stroke = ColorRGBa.fromHex(0x191970)

        drawer.circle(position, r)

        drawer.fill = ColorRGBa.RED
        drawer.stroke = ColorRGBa.RED
        drawer.text(node.name, position - Vector2(r, 0.0))

        for (parent in node.parents) {
            val index = indicesMap.getOrElse(parent) {
                throw Exception("Node is not listed in Bayesian Network: ${parent.name}")
            }

            drawArrow(i, index)
        }
    }
}

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

//    println("Risk high given traffic light is yellow: ${risk.probabilityTable[listOf("yellow", "high")]}")

    application {
        program {
            val font = FontImageMap.fromUrl("file:data/fonts/Aller_Bd.ttf", 16.0)
            extend {
                bayesian.sample.draw(drawer, font, bayesianNetwork)
            }
        }
    }
}