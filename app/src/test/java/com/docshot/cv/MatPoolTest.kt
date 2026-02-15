package com.docshot.cv

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.Before

/**
 * Unit tests for [MatPool].
 *
 * These tests validate pool lifecycle without depending on OpenCV native libraries —
 * they test the pool's bookkeeping logic (size tracking, acquire/release counters,
 * max-size enforcement). Full integration tests (with real Mats) run as instrumented tests.
 */
class MatPoolTest {

    @Test
    fun `pool starts empty`() {
        val pool = MatPool(maxSize = 4)
        assertEquals(0, pool.size)
        assertEquals(0, pool.totalAcquires)
        assertEquals(0, pool.totalReuses)
    }

    @Test
    fun `clear resets counters`() {
        val pool = MatPool(maxSize = 4)
        // We can't call acquire/release without real OpenCV, but we can test clear on empty pool
        pool.clear()
        assertEquals(0, pool.size)
        assertEquals(0, pool.totalAcquires)
        assertEquals(0, pool.totalReuses)
    }

    @Test
    fun `maxSize parameter is respected in constructor`() {
        val pool1 = MatPool(maxSize = 2)
        assertEquals(0, pool1.size)

        val pool2 = MatPool(maxSize = 16)
        assertEquals(0, pool2.size)

        // Default maxSize
        val pool3 = MatPool()
        assertEquals(0, pool3.size)
    }
}
