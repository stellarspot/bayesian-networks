package bayesian.core

import bayesian.util.initialMessage
import bayesian.util.multiplyMessageElementWise
import bayesian.util.multiplyTensorMessages

fun BayesianNetwork.loopyBeliefPropagation(vararg evidences: Evidence, maxSteps: Int = 100): Double {

    if (evidences.isEmpty()) {
        return 1.0
    }

    val graph = FactorGraph(this)
    graph.applyEvidences(*evidences)
    // graph.dump()
    graph.sendLoopyMessages(maxSteps, evidences.first())
    return graph.calculateMarginalization(*evidences)
}


fun FactorGraph.sendLoopyMessages(maxSteps: Int, evidence: Evidence, threshold: Double = 0.001) {

    val variables = variablesMap.values

    for (variable in variables) {
        for (edge in variable.outEdges) {
            edge.prevMessage = initialMessage(variable.domainSize)
            edge.message = initialMessage(variable.domainSize)
        }
    }


    var step = 0
    var prevMarginal = -1.0
    do {

        logger.debug("step: ${step++}")

        for (variable in variables) {
            for (edge in variable.outEdges) {
                sendLoopyMessage(variable, edge)
            }
        }

        for (factor in factors) {
            for ((index, edge) in factor.outEdges.withIndex()) {
                sendLoopyMessage(factor, edge, index)
            }
        }

        for (variable in variables) {
            for (edge in variable.inEdges + variable.outEdges) {
                edge.prevMessage = edge.message
            }
        }


        var marginal = calculateMarginalization(evidence)
        if (Math.abs(marginal - prevMarginal) <= threshold && marginal <= 1.0) {
            break
        }

        prevMarginal = marginal

    } while (step < maxSteps)
}

private fun FactorGraph.sendLoopyMessage(variable: VariableNode, edge: Edge) {
    val edges = variable.outEdges
    if (edges.size == 1) {
    } else {

        val messages = variable
                .inEdges
                .filter { it.factor != edge.factor }
                .map { it.prevMessage }
                .filterNotNull()

        var message = initialMessage(variable.domainSize)
        for (msg in messages) {
            message = multiplyMessageElementWise(message, msg)
        }
        edge.message = message
        logger.debug("send message(v->f): $variable, ${edge.factor}:\n${edge.message}")
    }
}

private fun FactorGraph.sendLoopyMessage(factor: FactorNode, edge: Edge, index: Int) {

    val edges = factor.outEdges
    if (edges.size == 1) {
        edge.message = factor.tensor
        logger.debug("send message(f->v): $factor, ${edge.variable}, initial: ${edge.message}")
    } else {

        val messages = factor.inEdges.map { it.prevMessage }.toTypedArray()
        edge.message = multiplyTensorMessages(factor.tensor, index, *messages)
        logger.debug("send message(f->v): $factor, ${edge.variable}:\n${edge.message}")
    }
}