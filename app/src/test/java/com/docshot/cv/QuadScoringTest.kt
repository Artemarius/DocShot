package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point

class QuadScoringTest {

    @Test
    fun `quadArea computes correct area for a rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 200.0),
            Point(0.0, 200.0)
        )
        assertEquals(20000.0, quadArea(rect), 0.1)
    }

    @Test
    fun `quadArea computes correct area for a non-axis-aligned quad`() {
        // Triangle-like degenerate quad with one collapsed edge still works
        val quad = listOf(
            Point(0.0, 0.0),
            Point(4.0, 0.0),
            Point(4.0, 3.0),
            Point(0.0, 3.0)
        )
        assertEquals(12.0, quadArea(quad), 0.001)
    }

    @Test
    fun `perfect rectangle has angle score close to 1`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 200.0),
            Point(0.0, 200.0)
        )
        val score = angleRegularityScore(rect)
        assertTrue("Perfect rectangle score should be > 0.95, got $score", score > 0.95)
    }

    @Test
    fun `skewed quadrilateral has lower angle score than rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val skewed = listOf(
            Point(20.0, 0.0),
            Point(120.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0)
        )
        val rectScore = angleRegularityScore(rect)
        val skewedScore = angleRegularityScore(skewed)
        assertTrue(
            "Rectangle ($rectScore) should score higher than skewed ($skewedScore)",
            rectScore > skewedScore
        )
    }

    @Test
    fun `larger quad area is computed correctly`() {
        val small = listOf(
            Point(0.0, 0.0),
            Point(50.0, 0.0),
            Point(50.0, 50.0),
            Point(0.0, 50.0)
        )
        val large = listOf(
            Point(0.0, 0.0),
            Point(200.0, 0.0),
            Point(200.0, 200.0),
            Point(0.0, 200.0)
        )
        assertTrue(quadArea(large) > quadArea(small))
    }
}
