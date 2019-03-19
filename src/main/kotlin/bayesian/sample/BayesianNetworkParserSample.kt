package bayesian.sample

import bayesian.draw.openrndr.draw
import bayesian.parser.parse
import java.io.File

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println("Provide path to Bayesian Network file with *.bif extension")
        System.exit(1)
    }
    val file = args[0]

    println("file path: $file")

    val network = parse(File(file).toURL())
    draw(network, "Parsed Bayesian Network", 1000, 600, 30)
}