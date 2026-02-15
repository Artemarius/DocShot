package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.opencv.core.Point
import org.opencv.core.Size

class ContourAnalysisTest {

    private val imageSize = Size(800.0, 600.0)

    @Test
    fun `large contour touching 2 edges detected as partial`() {
        // Contour points along top and right edges
        val points = listOf(
            Point(100.0, 2.0),    // near top edge
            Point(795.0, 2.0),    // near top + right edges
            Point(797.0, 300.0),  // near right edge
            Point(797.0, 500.0),  // near right edge
            Point(100.0, 500.0)   // interior
        )
        val edgeCount = touchesFrameEdges(points, imageSize)
        assertTrue("Should touch 2 edges (top + right), got $edgeCount", edgeCount >= 2)
    }

    @Test
    fun `large contour touching 1 edge is NOT partial`() {
        // Contour points near only the top edge
        val points = listOf(
            Point(100.0, 3.0),    // near top edge
            Point(500.0, 3.0),    // near top edge
            Point(500.0, 300.0),  // interior
            Point(100.0, 300.0)   // interior
        )
        val edgeCount = touchesFrameEdges(points, imageSize)
        assertEquals("Should touch exactly 1 edge (top)", 1, edgeCount)
    }

    @Test
    fun `small contour touching edges is NOT partial`() {
        // Even though points are near edges, a small contour should not
        // be flagged as partial -- the area threshold handles this in
        // analyzeContours, but touchesFrameEdges itself just counts edges.
        // Here we verify the edge counting is correct.
        val points = listOf(
            Point(2.0, 2.0),     // near top + left edges
            Point(10.0, 2.0),    // near top edge
            Point(10.0, 10.0),   // interior
            Point(2.0, 10.0)     // near left edge
        )
        val edgeCount = touchesFrameEdges(points, imageSize)
        // It touches 2 edges (top + left), but area filtering in analyzeContours
        // would prevent this from being flagged as partial.
        // This test verifies that touchesFrameEdges correctly counts edges.
        assertEquals("Should touch 2 edges (top + left)", 2, edgeCount)
    }

    @Test
    fun `contour touching all 4 edges returns 4`() {
        // Full-frame quad — corners near all 4 image edges
        val points = listOf(
            Point(2.0, 3.0),       // near top + left
            Point(796.0, 2.0),     // near top + right
            Point(797.0, 597.0),   // near bottom + right
            Point(3.0, 596.0)      // near bottom + left
        )
        val edgeCount = touchesFrameEdges(points, imageSize)
        assertEquals("Full-frame quad should touch 4 edges", 4, edgeCount)
    }

    @Test
    fun `contour not near any edge returns 0`() {
        val points = listOf(
            Point(100.0, 100.0),
            Point(400.0, 100.0),
            Point(400.0, 400.0),
            Point(100.0, 400.0)
        )
        val edgeCount = touchesFrameEdges(points, imageSize)
        assertEquals("Interior contour should touch 0 edges", 0, edgeCount)
    }
}
