package bayesian.draw.openrndr

import bayesian.core.BayesianNetwork
import bayesian.core.Node
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.draw.Drawer
import org.openrndr.draw.FontImageMap
import org.openrndr.draw.FontMap
import org.openrndr.math.Vector2
import java.lang.Exception

fun draw(bayesianNetwork: BayesianNetwork) {
    application {
        program {
            val font = FontImageMap.fromUrl("file:data/fonts/Aller_Bd.ttf", 16.0)
            extend {
                draw(drawer, font, bayesianNetwork)
            }
        }
    }
}

private fun draw(drawer: Drawer, font: FontMap, network: BayesianNetwork) {

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