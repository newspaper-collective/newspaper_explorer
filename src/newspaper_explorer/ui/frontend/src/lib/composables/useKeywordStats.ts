/**
 * Keyword statistics composable for analysis views
 * 
 * Handles loading statistics from backend and generating chart configurations
 * Similar to useLayoutStats but for keyword data
 */

import { ref, computed } from 'vue'
import type { EChartsOption } from 'echarts'

export interface KeywordStats {
  total?: number
  total_occurrences?: number
  avg_score?: number
  avg_frequency?: number
  unique_documents?: number
  score_distribution?: {
    bins: string[]
    counts: number[]
  }
  top_keywords?: Array<{
    keyword: string
    frequency: number
    score: number
  }>
  keywords_per_doc?: number[]
}

export function useKeywordStats() {
  const backendStats = ref<KeywordStats | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const statistics = computed(() => {
    if (!backendStats.value) return null

    return {
      totalKeywords: backendStats.value.total || 0,
      totalOccurrences: backendStats.value.total_occurrences || 0,
      avgScore: backendStats.value.avg_score || 0,
      avgFrequency: backendStats.value.avg_frequency || 0,
      documentsWithKeywords: backendStats.value.unique_documents || 0,
    }
  })

  /**
   * Create top keywords bar chart from stats data
   */
  function createTopKeywordsChart(
    topKeywords: Array<{ keyword: string; frequency: number; score: number }>,
    count: number
  ): EChartsOption {
    if (!topKeywords || topKeywords.length === 0) {
      return {}
    }

    const data = topKeywords.slice(0, count)

    return {
      title: {
        text: `Top ${data.length} Keywords by Frequency`,
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        formatter: (params: any) => {
          const item = params[0]
          const kw = data.find((k) => k.keyword === item.name)
          return `${item.name}<br/>Frequency: <strong>${item.value}</strong><br/>Score: <strong>${
            kw ? kw.score.toFixed(3) : 'N/A'
          }</strong>`
        },
      },
      xAxis: {
        type: 'category',
        data: data.map((k) => k.keyword),
        axisLabel: {
          rotate: 45,
          interval: 0,
        },
      },
      yAxis: {
        type: 'value',
        name: 'Frequency',
      },
      series: [
        {
          type: 'bar',
          data: data.map((k) => k.frequency),
          itemStyle: { color: '#5470c6' },
        },
      ],
      grid: { bottom: 120 },
    }
  }

  /**
   * Create score distribution chart from stats data
   */
  function createScoreDistributionChart(scoreDistribution: {
    bins: string[]
    counts: number[]
  }): EChartsOption {
    if (!scoreDistribution || !scoreDistribution.counts) {
      return {}
    }

    return {
      title: {
        text: 'TF-IDF Score Distribution',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      xAxis: {
        type: 'category',
        data: scoreDistribution.bins,
        axisLabel: { rotate: 45, interval: 2 },
      },
      yAxis: {
        type: 'value',
        name: 'Count',
      },
      series: [
        {
          type: 'bar',
          data: scoreDistribution.counts,
          itemStyle: { color: '#5470c6' },
        },
      ],
      grid: { bottom: 80 },
    }
  }

  /**
   * Create keywords per document distribution chart
   */
  function createKeywordsPerDocChart(keywordsPerDoc: number[]): EChartsOption {
    if (!keywordsPerDoc || keywordsPerDoc.length === 0) {
      return {}
    }

    // Use reduce to find max instead of spread operator
    const maxKw = keywordsPerDoc.reduce((max, val) => Math.max(max, val), 0)
    const docBins = Array(Math.min(maxKw + 1, 20)).fill(0)

    keywordsPerDoc.forEach((count) => {
      const binIndex = Math.min(count, docBins.length - 1)
      docBins[binIndex]++
    })

    const xAxisData = []
    for (let i = 0; i < docBins.length; i++) {
      xAxisData.push(i.toString())
    }

    return {
      title: {
        text: 'Keywords per Document Distribution',
        left: 'center',
        textStyle: { fontSize: 16, fontWeight: 'normal' },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        name: 'Keywords per Document',
      },
      yAxis: {
        type: 'value',
        name: 'Document Count',
      },
      series: [
        {
          type: 'bar',
          data: docBins,
          itemStyle: { color: '#91cc75' },
        },
      ],
    }
  }

  /**
   * Create wordcloud from top keywords
   */
  function createWordcloudData(
    topKeywords: Array<{ keyword: string; frequency: number }>,
    limit: number
  ): Array<{ name: string; value: number }> {
    if (!topKeywords || topKeywords.length === 0) {
      return []
    }

    return topKeywords.slice(0, limit).map((k) => ({
      name: k.keyword,
      value: k.frequency,
    }))
  }

  return {
    backendStats,
    statistics,
    loading,
    error,
    createTopKeywordsChart,
    createScoreDistributionChart,
    createKeywordsPerDocChart,
    createWordcloudData,
  }
}
