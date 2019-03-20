package bayesian.message

import bayesian.util.*
import org.junit.Assert
import org.junit.Test
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import kotlin.test.assertEquals

class MessageUtilTest {

    @Test
    fun testInitialMessage() {
        assertTensorEquals(
                initialMessage(3),
                Nd4j.create(floatArrayOf(1f, 1f, 1f), intArrayOf(3)))
    }

    @Test
    fun testTransposeLastAxisToFirst() {

        assertTensorEquals(
                transposeLastAxisToFirst(Nd4j.create(floatArrayOf(1f, 2f, 3f), intArrayOf(3, 1))),
                Nd4j.create(floatArrayOf(1f, 2f, 3f), intArrayOf(1, 3)))

        assertTensorEquals(
                transposeLastAxisToFirst(Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), intArrayOf(3, 2))),
                Nd4j.create(floatArrayOf(1f, 3f, 5f, 2f, 4f, 6f), intArrayOf(2, 3)))
    }

    @Test
    fun testMessageMultiplication1() {

        val tensor = Nd4j.create(floatArrayOf(1f, 2f, 3f), intArrayOf(3, 1))
        val message = Nd4j.create(floatArrayOf(2f), intArrayOf(1))

        assertTensorEquals(
                multiplyTensorMessage(tensor, message),
                Nd4j.create(floatArrayOf(2f, 4f, 6f), intArrayOf(3)))
    }

    @Test
    fun testMessageMultiplication2() {

        val tensor = Nd4j.create(floatArrayOf(1f, 2f, 3f), intArrayOf(1, 3))
        val message = Nd4j.create(floatArrayOf(1f, 2f, 3f), intArrayOf(3))

        assertTensorEquals(
                multiplyTensorMessage(tensor, message),
                Nd4j.create(floatArrayOf(14f), intArrayOf(1)))
    }

    @Test
    fun testMessageMultiplication3() {

        val tensor2D = Nd4j.create(
                floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f),
                intArrayOf(3, 2))

        val tensor = takeTensor(tensor2D, 1, 0)
        val message = initialMessage(1)

        assertTensorEquals(
                multiplyTensorMessage(tensor, message),
                Nd4j.create(floatArrayOf(1f, 3f, 5f), intArrayOf(3)))
    }
}

fun assertTensorEquals(expected: INDArray, actual: INDArray) {
    assertEquals(expected, actual)
    Assert.assertArrayEquals(expected.shape(), actual.shape())
}