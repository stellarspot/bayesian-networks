package bayesian.message

import bayesian.util.*
import org.junit.Ignore
import org.junit.Test
import org.nd4j.linalg.factory.Nd4j

class TensorRiskTrafficLightTest {

    @Test
    fun test() {

        // P(Risk, TrafficLight) = P1(Risk|TrafficLight) * P2 (TrafficLight)
        // Factor Graph
        // Risk - P1 - Traffic Light - P2

        val trafficLightTensor = Nd4j.create(
                floatArrayOf(0.4f, 0.25f, 0.35f),
                intArrayOf(3))

        val riskTensor = Nd4j.create(
                floatArrayOf(0.1f, 0.9f, 0.55f, 0.45f, 0.95f, 0.05f),
                intArrayOf(3, 2))

        val riskIndex = 1
        val riskEvidenceIndex = 0

        val riskEvidenceTensor = takeTensor(riskTensor, riskIndex, riskEvidenceIndex)

        assertTensorEquals(
                Nd4j.create(floatArrayOf(0.1f, 0.55f, 0.95f), intArrayOf(3, 1)),
                riskEvidenceTensor)


        // Risk -> P1
        val messageRiskP1 = initialMessage(1)
        // P1 -> TrafficLight
        val messageP1TrafficLight = multiplyMessage(riskEvidenceTensor, messageRiskP1)

        assertTensorEquals(
                Nd4j.create(floatArrayOf(0.1f, 0.55f, 0.95f), intArrayOf(3)),
                messageP1TrafficLight)

        // P2 -> TrafficLight
        val messageP2TrafficLight = trafficLightTensor

        // TrafficLight -> P1
        val messageTrafficLightP1 = messageP2TrafficLight

        // P1 -> Risk
        val messageP1Risk = multiplyMessage(
                transposeLastAxisToFirst(riskEvidenceTensor),
                messageTrafficLightP1)

        assertTensorEquals(
                Nd4j.create(floatArrayOf(0.51f), intArrayOf(1)),
                messageP1Risk
        )
    }
}