package bayesian.util

import org.junit.Assert
import java.net.URL

private val epsilon = 0.01


fun assertDoubleEquals(expected: Double, actual: Double) {
    Assert.assertEquals(expected, actual, epsilon)
}

fun getResourceUrl(resource: String): URL? {
    return TestUtil::javaClass.javaClass.classLoader.getResource(resource)
}

class TestUtil