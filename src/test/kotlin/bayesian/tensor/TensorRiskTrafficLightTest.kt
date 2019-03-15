package bayesian.tensor

import bayesian.util.takeTensor
import org.junit.Assert
import org.junit.Assert.assertArrayEquals
import org.junit.Test
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertEquals

class TensorRiskTrafficLightTest {

    @Test
    fun test() {


        val trafficLightTensor = Nd4j.create(
                floatArrayOf(0.4f, 0.25f, 0.35f),
                intArrayOf(3))

        val riskTensor = Nd4j.create(
                floatArrayOf(0.1f, 0.9f, 0.5f, 0.45f, 0.95f, 0.05f),
                intArrayOf(3, 2))

        val riskIndex = 1
        val riskEvidenceIndex = 0

        val riskEvidenceTensor = takeTensor(riskTensor, riskIndex, riskEvidenceIndex)
        val expectedRiskEvidenceTensor = Nd4j.create(floatArrayOf(0.1f, 0.5f, 0.95f), intArrayOf(3, 1))

        assertTensorEquals(expectedRiskEvidenceTensor, riskEvidenceTensor)

        println("risk evidence shape: ${riskEvidenceTensor.shape().contentToString()}")
        println("risk evidence tensor:\n${riskEvidenceTensor}")

        val riskP1Message = Nd4j.ones(1)
        println("message (risk->P1): ${riskP1Message.shape().contentToString()}")

        val messageP1TrafficLight = riskEvidenceTensor.mul(riskP1Message)
        println("message (P1->TL):\n$messageP1TrafficLight")

    }

    fun assertTensorEquals(expected: INDArray, actual: INDArray) {
        assertEquals(expected, actual)
        assertArrayEquals(expected.shape(), actual.shape())
    }
}