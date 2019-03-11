package bayesian.sample

import org.openrndr.application
import org.openrndr.color.ColorRGBa

fun main(args: Array<String>) {
    application {
        configure {
            width = 600
            height = 400
        }
        program {
            extend {
                drawer.background(ColorRGBa.PINK)
                drawer.fill = ColorRGBa.WHITE
                drawer.circle(drawer.bounds.center, 100.0)
            }
        }
    }
}