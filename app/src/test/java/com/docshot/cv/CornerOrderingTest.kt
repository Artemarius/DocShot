package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Test
import org.opencv.core.Point

class CornerOrderingTest {

    @Test
    fun `already ordered corners remain unchanged`() {
        val tl = Point(10.0, 10.0)
        val tr = Point(200.0, 10.0)
        val br = Point(200.0, 300.0)
        val bl = Point(10.0, 300.0)

        val result = orderCorners(listOf(tl, tr, br, bl))
        assertCornersEqual(listOf(tl, tr, br, bl), result)
    }

    @Test
    fun `reversed corners are reordered correctly`() {
        val tl = Point(10.0, 10.0)
        val tr = Point(200.0, 10.0)
        val br = Point(200.0, 300.0)
        val bl = Point(10.0, 300.0)

        val result = orderCorners(listOf(br, bl, tl, tr))
        assertCornersEqual(listOf(tl, tr, br, bl), result)
    }

    @Test
    fun `scrambled corners are reordered correctly`() {
        val tl = Point(50.0, 30.0)
        val tr = Point(400.0, 20.0)
        val br = Point(420.0, 350.0)
        val bl = Point(30.0, 360.0)

        val result = orderCorners(listOf(bl, tr, tl, br))
        assertCornersEqual(listOf(tl, tr, br, bl), result)
    }

    @Test
    fun `skewed quadrilateral corners are ordered correctly`() {
        // Document photographed at a steep angle
        val tl = Point(120.0, 80.0)
        val tr = Point(500.0, 60.0)
        val br = Point(540.0, 420.0)
        val bl = Point(80.0, 400.0)

        val result = orderCorners(listOf(br, tl, bl, tr))
        assertCornersEqual(listOf(tl, tr, br, bl), result)
    }

    @Test
    fun `nearly square corners`() {
        val tl = Point(0.0, 0.0)
        val tr = Point(100.0, 0.0)
        val br = Point(100.0, 100.0)
        val bl = Point(0.0, 100.0)

        val result = orderCorners(listOf(bl, br, tl, tr))
        assertCornersEqual(listOf(tl, tr, br, bl), result)
    }

    private fun assertCornersEqual(expected: List<Point>, actual: List<Point>) {
        assertEquals(4, actual.size)
        for (i in expected.indices) {
            assertEquals("Corner $i x", expected[i].x, actual[i].x, 0.001)
            assertEquals("Corner $i y", expected[i].y, actual[i].y, 0.001)
        }
    }
}
