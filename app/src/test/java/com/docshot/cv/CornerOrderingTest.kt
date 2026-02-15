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

    @Test
    fun `diamond (45 degree rotation) corners are ordered correctly`() {
        // Document rotated ~45° — the sum/diff method previously assigned
        // the same point to two roles, producing a degenerate triangle overlay.
        //       T(100,0)
        //      / \
        // L(0,100)  R(200,100)
        //      \ /
        //       B(100,200)
        val top = Point(100.0, 0.0)
        val right = Point(200.0, 100.0)
        val bottom = Point(100.0, 200.0)
        val left = Point(0.0, 100.0)

        val result = orderCorners(listOf(top, right, bottom, left))
        // CW from TL: top → right → bottom → left (proper non-crossing quad)
        assertCornersEqual(listOf(top, right, bottom, left), result)

        // Verify all 4 points are distinct (no duplicates)
        val unique = result.map { Pair(it.x, it.y) }.toSet()
        assertEquals("All 4 corners must be distinct points", 4, unique.size)
    }

    @Test
    fun `near-45-degree rotation corners are ordered correctly`() {
        // Slightly off from perfect 45° — still triggers near-tie in y-x
        val tl = Point(90.0, 10.0)
        val tr = Point(190.0, 95.0)
        val br = Point(110.0, 195.0)
        val bl = Point(10.0, 110.0)

        val result = orderCorners(listOf(br, tl, tr, bl))
        assertCornersEqual(listOf(tl, tr, br, bl), result)

        val unique = result.map { Pair(it.x, it.y) }.toSet()
        assertEquals("All 4 corners must be distinct points", 4, unique.size)
    }

    private fun assertCornersEqual(expected: List<Point>, actual: List<Point>) {
        assertEquals(4, actual.size)
        for (i in expected.indices) {
            assertEquals("Corner $i x", expected[i].x, actual[i].x, 0.001)
            assertEquals("Corner $i y", expected[i].y, actual[i].y, 0.001)
        }
    }
}
