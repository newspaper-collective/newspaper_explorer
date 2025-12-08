/**
 * Layout statistics composable for analysis views
 *
 * Handles loading statistics from backend and generating chart configurations
 * Used in PicturesView, LayoutView, and other analysis views
 */

import { ref, computed } from 'vue'
import type { EChartsOption } from 'echarts'
import { getResultColor } from '@/lib/colors'

export interface LayoutStats {
  total?: number
  unique_pages?: number
  unique_classes?: number
  avg_confidence?: number
  avg_width?: number
  avg_height?: number
  timeline?: Array<{ date: string; value: number }>
  confidence_distribution?: {
    bins: string[]
    counts: number[]
  }
  position_distribution?: Array<{
    bin_start: number
    count: number
  }>
  size_distribution?: Array<[number, number]>
}

export interface StatsOptions {
  label?: string
  min_confidence?: number
  run_id?: string | null
  [key: string]: any
}

export function useLayoutStats() {
  const backendStats = ref<LayoutStats | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const statistics = computed(() => {
    if (!backendStats.value) return null

    return {
      totalPictures: backendStats.value.total || 0,
      uniquePages: backendStats.value.unique_pages || 0,
      avgConfidence: backendStats.value.avg_confidence || 0,
      avgWidth: backendStats.value.avg_width || 0,
      avgHeight: backendStats.value.avg_height || 0,
    }
  })

  /**
   * Create timeline chart from stats data
   */
  function createTimelineChart(timelineData: Array<{ date: string; value: number }>): EChartsOption {
    if (!timelineData || timelineData.length === 0) {
      return {}
    }

    // Sort by date and prepare data
    const sortedData = [...timelineData].sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    )

    const dates = sortedData.map((d) => d.date)
    const values = sortedData.map((d) => d.value)

    return {
      title: {
        text: 'Pictures Over Time',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'line' },
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          rotate: 45,
          formatter: (value: string) => {
            const date = new Date(value)
            return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short' })
          },
        },
      },
      yAxis: {
        type: 'value',
        name: 'Count',
      },
      series: [
        {
          name: 'Pictures',
          type: 'line',
          data: values,
          smooth: true,
          itemStyle: { color: getResultColor(1) },
          areaStyle: { opacity: 0.3 },
        },
      ],
      grid: { bottom: 80, left: 60, right: 40, top: 60 },
    }
  }

  /**
   * Create confidence distribution histogram
   */
  function createConfidenceChart(distribution: { bins: string[]; counts: number[] }): EChartsOption {
    const { bins, counts } = distribution

    return {
      title: {
        text: 'Confidence Distribution',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      xAxis: {
        type: 'category',
        data: bins,
        axisLabel: { rotate: 45, fontSize: 10 },
        name: 'Confidence',
      },
      yAxis: {
        type: 'value',
        name: 'Count',
      },
      series: [
        {
          type: 'bar',
          data: counts,
          itemStyle: { color: getResultColor(1) },
        },
      ],
      grid: { bottom: 80 },
    }
  }

  /**
   * Create position distribution chart
   */
  function createPositionChart(
    distribution: Array<{ bin_start: number; count: number }>
  ): EChartsOption {
    const posData = distribution.map((d) => ({
      name: `${d.bin_start}-${d.bin_start + 200}`,
      value: d.count,
    }))

    return {
      title: {
        text: 'Vertical Position Distribution',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const data = params[0]
          return `Y: ${data.name}px<br/>Count: <strong>${data.value}</strong>`
        },
      },
      xAxis: {
        type: 'category',
        data: posData.map((d) => d.name),
        name: 'Y Position (px)',
        axisLabel: { fontSize: 10, rotate: 45 },
      },
      yAxis: {
        type: 'value',
        name: 'Count',
      },
      series: [
        {
          type: 'bar',
          data: posData.map((d) => d.value),
          itemStyle: { color: getResultColor(3) },
        },
      ],
      grid: { bottom: 80 },
    }
  }

  /**
   * Create size distribution scatter plot
   */
  function createSizeChart(distribution: Array<[number, number]>): EChartsOption {
    return {
      title: {
        text: 'Picture Size Distribution',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          return `Width: ${params.value[0].toFixed(0)}<br/>Height: ${params.value[1].toFixed(0)}`
        },
      },
      xAxis: {
        type: 'value',
        name: 'Width (px)',
      },
      yAxis: {
        type: 'value',
        name: 'Height (px)',
      },
      series: [
        {
          type: 'scatter',
          data: distribution,
          itemStyle: { color: getResultColor(5), opacity: 0.6 },
          symbolSize: 6,
        },
      ],
    }
  }

  /**
   * Generate all chart configurations from stats
   */
  function generateCharts(stats: LayoutStats) {
    const charts: Record<string, EChartsOption> = {}

    if (stats.timeline) {
      charts.timeline = createTimelineChart(stats.timeline)
    }

    if (stats.confidence_distribution) {
      charts.confidence = createConfidenceChart(stats.confidence_distribution)
    }

    if (stats.position_distribution) {
      charts.position = createPositionChart(stats.position_distribution)
    }

    if (stats.size_distribution) {
      charts.size = createSizeChart(stats.size_distribution)
    }

    return charts
  }

  return {
    // State
    backendStats,
    statistics,
    loading,
    error,

    // Chart generators
    createTimelineChart,
    createConfidenceChart,
    createPositionChart,
    createSizeChart,
    generateCharts,
  }
}
