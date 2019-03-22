package bayesian.core

import bayesian.util.*
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory


fun BayesianNetwork.beliefPropagation(vararg evidences: Evidence): Double {

    if (evidences.isEmpty()) {
        return 1.0
    }

    val graph = FactorGraph(this)
    graph.applyEvidences(*evidences)
    // graph.dump()
    graph.sendMessages()
    return graph.calculateMarginalization(*evidences)
}

fun FactorGraph.sendMessages() {

    val variables = variablesMap.values
    var step = 0
    do {

        var finished = true
        logger.debug("step: ${step++}")

        for (variable in variables) {
            for (edge in variable.outEdges) {
                if (edge.message == null) {
                    finished = false
                    sendMessage(variable, edge)
                }
            }
        }

        for (factor in factors) {
            for ((index, edge) in factor.outEdges.withIndex()) {
                if (edge.message == null) {
                    finished = false
                    sendMessage(factor, edge, index)
                }
            }
        }

    } while (!finished)
}

private fun FactorGraph.sendMessage(variable: VariableNode, edge: Edge) {
    val edges = variable.outEdges
    if (edges.size == 1) {
        edge.message = initialMessage(variable.domainSize)
        logger.debug("send message(v->f): $variable, ${edge.factor}, initial ${edge.message}")
    } else {

        val inEdges = variable.inEdges.filter { it.factor != edge.factor }
        val canSendMessage = inEdges.all { it.message != null }

        if (canSendMessage) {
            val messages = inEdges
                    .map { it.message }
                    .filterNotNull()
            assert(messages.size == inEdges.size)

            var message = initialMessage(variable.domainSize)
            for (msg in messages) {
                message = multiplyMessageElementWise(message, msg)
            }
            edge.message = message
            logger.debug("send message(v->f): $variable, ${edge.factor}:\n${edge.message}")
        }
    }
}

private fun FactorGraph.sendMessage(factor: FactorNode, edge: Edge, index: Int) {

    val edges = factor.outEdges
    if (edges.size == 1) {
        edge.message = factor.tensor
        logger.debug("send message(f->v): $factor, ${edge.variable}, initial: ${edge.message}")
    } else {

        val inEdges = factor.inEdges

        val canSendMessage = inEdges
                .filterIndexed { i, _ -> i != index }
                .all { it.message != null }

        if (canSendMessage) {

            val messages = inEdges.map { it.message }.toTypedArray()
            edge.message = multiplyTensorMessages(factor.tensor, index, *messages)
            logger.debug("send message(f->v): $factor, ${edge.variable}:\n${edge.message}")
        }

    }
}