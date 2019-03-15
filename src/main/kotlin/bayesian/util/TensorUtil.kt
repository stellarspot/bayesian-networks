package bayesian.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun takeTensor(tensor: INDArray,
               index: Int,
               evidenceIndex: Int): INDArray {

    val size = tensor.shape().size
    if (size == 1) {
        val value = tensor.getDouble(evidenceIndex)
        return Nd4j.create(doubleArrayOf(value))
    }

    val dimensions = (0 until size).filter { it != index }.toIntArray()
    val subTensor = tensor.tensorAlongDimension(evidenceIndex, *dimensions)
    val reshapeDimension = tensor.shape().mapIndexed { i, dimension ->
        if (i == index) 1 else dimension.toInt()
    }.toIntArray()

    return subTensor.reshape(reshapeDimension)
}