package com.docshot.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.BeforeClass
import org.junit.Test
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point

class CornerTrackerTest {

    companion object {
        private var openCvLoaded = false

        @JvmStatic
        @BeforeClass
        fun loadOpenCV() {
            openCvLoaded = try {
                System.loadLibrary("opencv_java4")
                true
            } catch (_: UnsatisfiedLinkError) {
                try {
                    System.loadLibrary("opencv_java412")
                    true
                } catch (_: UnsatisfiedLinkError) {
                    false
                }
            }
        }
    }

    // ── Pure math tests (no OpenCV native required) ─────────────────────

    // -- isConvex --

    @Test
    fun `isConvex returns true for axis-aligned rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0)
        )
        assertTrue(CornerTracker.isConvex(rect))
    }

    @Test
    fun `isConvex returns true for convex trapezoid`() {
        val trap = listOf(
            Point(20.0, 0.0), Point(80.0, 0.0),
            Point(100.0, 60.0), Point(0.0, 60.0)
        )
        assertTrue(CornerTracker.isConvex(trap))
    }

    @Test
    fun `isConvex returns false for concave quad`() {
        // Push one corner inward to make it non-convex
        val concave = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(50.0, 30.0), // pushed inward
            Point(0.0, 80.0)
        )
        assertFalse(CornerTracker.isConvex(concave))
    }

    @Test
    fun `isConvex returns false for self-intersecting (bowtie) quad`() {
        // Swap two corners to create a bowtie
        val bowtie = listOf(
            Point(0.0, 0.0), Point(100.0, 80.0),
            Point(100.0, 0.0), Point(0.0, 80.0)
        )
        assertFalse(CornerTracker.isConvex(bowtie))
    }

    @Test
    fun `isConvex accepts CW and CCW winding`() {
        val cw = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0)
        )
        val ccw = cw.reversed()
        assertTrue(CornerTracker.isConvex(cw))
        assertTrue(CornerTracker.isConvex(ccw))
    }

    // -- quadArea --

    @Test
    fun `quadArea computes correct area for rectangle`() {
        val rect = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 50.0), Point(0.0, 50.0)
        )
        assertEquals(5000.0, CornerTracker.quadArea(rect), 0.01)
    }

    @Test
    fun `quadArea computes correct area for square`() {
        val sq = listOf(
            Point(10.0, 10.0), Point(110.0, 10.0),
            Point(110.0, 110.0), Point(10.0, 110.0)
        )
        assertEquals(10000.0, CornerTracker.quadArea(sq), 0.01)
    }

    @Test
    fun `quadArea is independent of winding order`() {
        val cw = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 50.0), Point(0.0, 50.0)
        )
        val ccw = cw.reversed()
        assertEquals(CornerTracker.quadArea(cw), CornerTracker.quadArea(ccw), 0.01)
    }

    @Test
    fun `quadArea handles non-rectangular quad`() {
        // Trapezoid: top edge 60px, bottom edge 100px, height 50px
        // Area = (60 + 100) / 2 * 50 = 4000
        val trap = listOf(
            Point(20.0, 0.0), Point(80.0, 0.0),
            Point(100.0, 50.0), Point(0.0, 50.0)
        )
        assertEquals(4000.0, CornerTracker.quadArea(trap), 0.01)
    }

    // -- averageCornerDistance --

    @Test
    fun `averageCornerDistance is zero for identical corners`() {
        val corners = listOf(
            Point(10.0, 20.0), Point(100.0, 20.0),
            Point(100.0, 80.0), Point(10.0, 80.0)
        )
        assertEquals(0.0, CornerTracker.averageCornerDistance(corners, corners), 0.001)
    }

    @Test
    fun `averageCornerDistance computes correct value for uniform shift`() {
        val a = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0)
        )
        // Shift all corners by (3, 4) -> each distance = 5
        val b = a.map { Point(it.x + 3.0, it.y + 4.0) }
        assertEquals(5.0, CornerTracker.averageCornerDistance(a, b), 0.001)
    }

    @Test
    fun `averageCornerDistance averages across corners`() {
        val a = listOf(
            Point(0.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0)
        )
        // Move only corner 0 by 20px, rest by 0px -> average = 20/4 = 5
        val b = listOf(
            Point(20.0, 0.0), Point(100.0, 0.0),
            Point(100.0, 80.0), Point(0.0, 80.0)
        )
        assertEquals(5.0, CornerTracker.averageCornerDistance(a, b), 0.001)
    }

    // ── State machine tests (require OpenCV native) ─────────────────────

    private lateinit var tracker: CornerTracker

    @Before
    fun setUp() {
        if (openCvLoaded) {
            tracker = CornerTracker()
        }
    }

    private fun requireOpenCv() {
        assumeTrue("OpenCV native library not available", openCvLoaded)
    }

    /** Creates a synthetic grayscale image with a white rectangle on black background. */
    private fun createTestFrame(
        width: Int = 640,
        height: Int = 480,
        rectLeft: Int = 100,
        rectTop: Int = 80,
        rectRight: Int = 540,
        rectBottom: Int = 400
    ): Mat {
        val frame = Mat.zeros(height, width, CvType.CV_8UC1)
        // Draw a white filled rectangle (high-contrast feature for KLT)
        for (y in rectTop until rectBottom) {
            for (x in rectLeft until rectRight) {
                frame.put(y, x, byteArrayOf(255.toByte()))
            }
        }
        return frame
    }

    /** Corners of the rectangle drawn by createTestFrame. */
    private fun testCorners(
        rectLeft: Int = 100,
        rectTop: Int = 80,
        rectRight: Int = 540,
        rectBottom: Int = 400
    ): List<Point> = listOf(
        Point(rectLeft.toDouble(), rectTop.toDouble()),
        Point(rectRight.toDouble(), rectTop.toDouble()),
        Point(rectRight.toDouble(), rectBottom.toDouble()),
        Point(rectLeft.toDouble(), rectBottom.toDouble())
    )

    @Test
    fun `starts in DETECT_ONLY state`() {
        requireOpenCv()
        assertEquals(TrackingState.DETECT_ONLY, tracker.state)
    }

    @Test
    fun `transitions to TRACKING on high-confidence detection`() {
        requireOpenCv()
        val frame = createTestFrame()
        val corners = testCorners()

        val result = tracker.processFrame(frame, corners, 0.72)
        frame.release()

        assertEquals(TrackingState.TRACKING, result.state)
        assertNotNull(result.corners)
        assertFalse(result.isTracked) // first frame is from detection
    }

    @Test
    fun `stays in DETECT_ONLY on low-confidence detection`() {
        requireOpenCv()
        val frame = createTestFrame()
        val corners = testCorners()

        val result = tracker.processFrame(frame, corners, 0.40)
        frame.release()

        assertEquals(TrackingState.DETECT_ONLY, result.state)
        assertNotNull(result.corners) // pass-through detection
        assertFalse(result.isTracked)
    }

    @Test
    fun `stays in DETECT_ONLY on null detection`() {
        requireOpenCv()
        val frame = createTestFrame()

        val result = tracker.processFrame(frame, null, 0.0)
        frame.release()

        assertEquals(TrackingState.DETECT_ONLY, result.state)
        assertNull(result.corners)
    }

    @Test
    fun `tracks static corners across frames`() {
        requireOpenCv()
        val frame1 = createTestFrame()
        val frame2 = createTestFrame() // identical frame
        val corners = testCorners()

        // Frame 1: enter tracking
        tracker.processFrame(frame1, corners, 0.72)
        frame1.release()

        // Frame 2: KLT-only (not correction frame since trackingFrameCount=2, not divisible by 3)
        val result = tracker.processFrame(frame2, null, 0.0)
        frame2.release()

        assertEquals(TrackingState.TRACKING, result.state)
        assertTrue(result.isTracked)
        assertNotNull(result.corners)

        // Tracked corners should be very close to original (static scene)
        val drift = CornerTracker.averageCornerDistance(corners, result.corners!!)
        assertTrue("Drift should be < 1px for static scene, was ${drift}px", drift < 1.0)
    }

    @Test
    fun `tracks uniform translation`() {
        requireOpenCv()
        val frame1 = createTestFrame(rectLeft = 100, rectTop = 80, rectRight = 540, rectBottom = 400)
        val corners1 = testCorners(rectLeft = 100, rectTop = 80, rectRight = 540, rectBottom = 400)

        // Shift rectangle by 10px right
        val dx = 10
        val frame2 = createTestFrame(rectLeft = 100 + dx, rectTop = 80, rectRight = 540 + dx, rectBottom = 400)
        val expectedCorners = testCorners(rectLeft = 100 + dx, rectTop = 80, rectRight = 540 + dx, rectBottom = 400)

        tracker.processFrame(frame1, corners1, 0.72)
        frame1.release()

        val result = tracker.processFrame(frame2, null, 0.0)
        frame2.release()

        assertTrue(result.isTracked)
        assertNotNull(result.corners)

        // KLT should track the shift — allow some tolerance since corners are at edges
        val drift = CornerTracker.averageCornerDistance(expectedCorners, result.corners!!)
        assertTrue("Drift from expected should be < 5px, was ${drift}px", drift < 5.0)
    }

    @Test
    fun `resets to DETECT_ONLY on tracking failure with blank frame`() {
        requireOpenCv()
        val frame1 = createTestFrame()
        val corners = testCorners()

        tracker.processFrame(frame1, corners, 0.72)
        frame1.release()

        // Send a completely blank frame — KLT should fail (no features to track)
        val blank = Mat.zeros(480, 640, CvType.CV_8UC1)
        val result = tracker.processFrame(blank, null, 0.0)
        blank.release()

        assertEquals(TrackingState.DETECT_ONLY, result.state)
        assertFalse(result.isTracked)
    }

    @Test
    fun `resets on frame size mismatch`() {
        requireOpenCv()
        val frame1 = createTestFrame(width = 640, height = 480)
        val corners = testCorners()

        tracker.processFrame(frame1, corners, 0.72)
        frame1.release()

        // Different resolution frame
        val frame2 = Mat.zeros(240, 320, CvType.CV_8UC1)
        val result = tracker.processFrame(frame2, null, 0.0)
        frame2.release()

        assertEquals(TrackingState.DETECT_ONLY, result.state)
    }

    @Test
    fun `reset clears all state`() {
        requireOpenCv()
        val frame = createTestFrame()
        val corners = testCorners()

        tracker.processFrame(frame, corners, 0.72)
        frame.release()
        assertEquals(TrackingState.TRACKING, tracker.state)

        tracker.reset()

        assertEquals(TrackingState.DETECT_ONLY, tracker.state)
    }

    @Test
    fun `correction drift exceeding threshold resets tracking`() {
        requireOpenCv()
        val frame1 = createTestFrame()
        val corners1 = testCorners()

        // Enter tracking
        tracker.processFrame(frame1, corners1, 0.72)
        frame1.release()

        // Advance to a correction frame (frame 3 in tracking session)
        val frame2 = createTestFrame()
        tracker.processFrame(frame2, null, 0.0)
        frame2.release()

        // Frame 3 is a correction frame. Provide detection corners far from tracked position.
        val frame3 = createTestFrame()
        val farCorners = listOf(
            Point(200.0, 180.0), Point(440.0, 180.0),
            Point(440.0, 350.0), Point(200.0, 350.0)
        )
        val result = tracker.processFrame(frame3, farCorners, 0.72)
        frame3.release()

        // Drift > 8px should have reset tracking
        assertEquals(TrackingState.DETECT_ONLY, result.state)
        assertFalse(result.isTracked)
    }

    @Test
    fun `needsCorrectionDetection returns true on correction frames`() {
        requireOpenCv()
        val frame1 = createTestFrame()
        tracker.processFrame(frame1, testCorners(), 0.72)
        frame1.release()

        // After processFrame, trackingFrameCount = 1
        // Frame 2: trackingFrameCount will be 2 (not correction)
        // Frame 3: trackingFrameCount will be 3 (correction: 3 % 3 == 0)
        // needsCorrectionDetection is checked BEFORE processFrame increments the counter,
        // so we check after frame 1 (count=1) -> not correction
        // After frame 2 (count=2) -> not correction
        // After frame 3 (count=3) -> 3 % 3 == 0 -> correction

        // At count=1: not a correction frame
        assertFalse(tracker.needsCorrectionDetection())

        val frame2 = createTestFrame()
        tracker.processFrame(frame2, null, 0.0) // count becomes 2
        frame2.release()

        assertFalse(tracker.needsCorrectionDetection()) // 2 % 3 != 0

        val frame3 = createTestFrame()
        tracker.processFrame(frame3, null, 0.0) // count becomes 3
        frame3.release()

        assertTrue(tracker.needsCorrectionDetection()) // 3 % 3 == 0
    }
}
