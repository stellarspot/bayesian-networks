package bayesian.util

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

fun initialMessage(size: Int): INDArray = Nd4j.create(FloatArray(size) { 1f }, intArrayOf(size))

fun multiplyTensorMessage(tensor: INDArray, message: INDArray): INDArray {
    val tensorIndex = tensor.shape().size - 1
    val result = Nd4j.tensorMmul(tensor, message, arrayOf(intArrayOf(tensorIndex), intArrayOf(0)))
    return result.reshape(result.shape().dropLast(1).map { it.toInt() }.toIntArray())

}

fun multiplyTensorMessages(tensor: INDArray, index: Int, vararg messages: INDArray?): INDArray {

    var result = tensor

    for (i in (messages.size - 1 downTo 0)) {
        if (i == index) {
            if (i != 0) {
                result = transposeLastAxisToFirst(result)
            }
        } else {
            result = multiplyTensorMessage(result, messages[i]!!)
        }
    }

    return result
}

fun multiplyMessageElementWise(message1: INDArray, message2: INDArray): INDArray {
    return message1.mul(message2)
}

fun transposeLastAxisToFirst(tensor: INDArray): INDArray {
    val lastIndex = tensor.shape().size - 1
    val rearrange = (0..lastIndex)
            .map { if (it == 0) lastIndex else it - 1 }
            .toIntArray()
    return tensor.permute(*rearrange)
}

fun takeTensor(tensor: INDArray,
               index: Int,
               evidenceIndex: Int): INDArray {

    val size = tensor.shape().size
    if (size == 1) {
        val value = tensor.getDouble(evidenceIndex)
        return Nd4j.create(doubleArrayOf(value))
    }

    val dimensions = (0 until size).filter { it != index }.toIntArray()
    val subTensor = tensor.tensorAlongDimension(evidenceIndex, *dimensions).dup()

    val reshapeDimension = tensor.shape().mapIndexed { i, dimension ->
        if (i == index) 1 else dimension.toInt()
    }.toIntArray()

    return subTensor.reshape(reshapeDimension)
}

fun printlnTensor(msg: String, tensor: INDArray) {
    println("$msg shape : ${tensor.shape().contentToString()}\n$tensor")
}