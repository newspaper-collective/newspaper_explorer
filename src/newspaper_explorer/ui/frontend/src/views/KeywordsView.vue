<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import api from '@/lib/api'
import ResultsViewer from '@/components/ResultsViewer.vue'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import StatisticsCards from '@/components/StatisticsCards.vue'
import PaginationControls from '@/components/PaginationControls.vue'
import {
  useWordCloud,
  useTimelineChart,
} from '@/lib/charts'
import { usePagination } from '@/lib/composables/usePagination'
import { useKeywordStats } from '@/lib/composables/useKeywordStats'
import type { EChartsOption } from 'echarts'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
])

interface Keyword {
  keyword: string
  frequency: number
  tfidf_score: number
  doc_id?: string
}

interface KeywordDocument {
  doc_id: string
  date?: string
  score: number
  page_id?: string
}

interface CoOccurrence {
  keyword: string
  count: number
}

const sourceStore = useSourceStore()
const keywords = ref<Keyword[]>([])
const loading = ref(false)
const selectedRunId = ref<string | null>(null)
const resultsViewer = ref<InstanceType<typeof ResultsViewer>>()
const searchQuery = ref('')
const minScore = ref(0)
const topKeywordsCount = ref(20)
const wordcloudKeywordCount = ref(100)

// Pagination
const {
  currentPage,
  totalItems,
  totalPages,
  pageSize,
  goToPage,
} = usePagination({ itemsPerPage: 100 })

// Backend stats
const {
  backendStats,
  statistics,
  createTopKeywordsChart,
  createScoreDistributionChart,
  createKeywordsPerDocChart,
  createWordcloudData,
} = useKeywordStats()

// Keyword explorer state
const selectedKeyword = ref<string>('')
const keywordDocuments = ref<KeywordDocument[]>([])
const keywordCoOccurrences = ref<CoOccurrence[]>([])
const explorerLoading = ref(false)

// Chart options
const topKeywordsChart = ref<EChartsOption>({})
const wordcloudChart = ref<EChartsOption>({})
const timelineChart = ref<EChartsOption>({})
const scoreDistributionChart = ref<EChartsOption>({})
const keywordsPerDocChart = ref<EChartsOption>({})

// Statistics cards data
const statsCards = computed(() => {
  if (!statistics.value) return []
  return [
    { label: 'Documents', value: statistics.value.documentsWithKeywords, format: 'number' as const },
    { label: 'Total Keywords', value: statistics.value.totalKeywords, format: 'number' as const },
    { label: 'Total Occurrences', value: statistics.value.totalOccurrences, format: 'number' as const },
    { label: 'Avg Score', value: statistics.value.avgScore.toFixed(3), format: 'custom' as const },
    { label: 'Avg Frequency', value: statistics.value.avgFrequency.toFixed(1), format: 'custom' as const },
  ]
})

async function loadKeywords() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    // 1. Load statistics from backend (unfiltered for total overview)
    const unfilteredStatsParams: any = {
      min_score: 0, // No filter - show all keywords
    }
    if (selectedRunId.value) unfilteredStatsParams.run_id = selectedRunId.value

    const unfilteredStatsResponse = await api.get(
      `/keywords/${sourceStore.currentSource}/stats`,
      { params: unfilteredStatsParams }
    )

    // Store unfiltered stats for statistics cards
    backendStats.value = unfilteredStatsResponse.data

    // 2. Load filtered statistics for charts
    const filteredStatsParams: any = {
      min_score: minScore.value / 100,
    }
    if (selectedRunId.value) filteredStatsParams.run_id = selectedRunId.value

    const filteredStatsResponse = await api.get(
      `/keywords/${sourceStore.currentSource}/stats`,
      { params: filteredStatsParams }
    )

    updateChartsFromStats(filteredStatsResponse.data)

    // 3. Load paginated keywords for table display
    const params: any = {
      min_score: minScore.value / 100,
      page: currentPage.value,
      page_size: pageSize.value,
    }
    if (selectedRunId.value) params.run_id = selectedRunId.value
    if (searchQuery.value.trim()) params.search = searchQuery.value.trim()

    const response = await api.get(
      `/keywords/${sourceStore.currentSource}/`,
      { params }
    )

    // Handle both paginated and non-paginated responses
    if (response.data.items) {
      keywords.value = response.data.items
      totalItems.value = response.data.total
    } else {
      keywords.value = response.data
      totalItems.value = response.data.length
    }

    // Load timeline separately
    await loadTimeline()
  } catch (error) {
    console.error('Failed to load keywords:', error)
    keywords.value = []
    totalItems.value = 0
  } finally {
    loading.value = false
  }
}

function updateChartsFromStats(stats: any) {
  // Top keywords bar chart
  if (stats.top_keywords && stats.top_keywords.length > 0) {
    topKeywordsChart.value = createTopKeywordsChart(stats.top_keywords, topKeywordsCount.value)
  }

  // Score distribution
  if (stats.score_distribution) {
    scoreDistributionChart.value = createScoreDistributionChart(stats.score_distribution)
  }

  // Keywords per document distribution
  if (stats.keywords_per_doc && stats.keywords_per_doc.length > 0) {
    keywordsPerDocChart.value = createKeywordsPerDocChart(stats.keywords_per_doc)
  }

  // Wordcloud
  if (stats.top_keywords && stats.top_keywords.length > 0) {
    const wordcloudData = createWordcloudData(stats.top_keywords, wordcloudKeywordCount.value)
    wordcloudChart.value = useWordCloud(wordcloudData, {
      title: {
        text: `Keyword Word Cloud (${wordcloudKeywordCount.value} Words)`,
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'normal',
        },
      },
    })
  }
}

async function loadTimeline() {
  if (!sourceStore.currentSource) return

  try {
    const params: any = { aggregation: 'month', limit: 10 }
    if (selectedRunId.value) params.run_id = selectedRunId.value

    const response = await api.get(
      `/keywords/${sourceStore.currentSource}/timeline`,
      { params }
    )
    createTimelineChart(response.data)
  } catch (error) {
    console.error('Failed to load timeline:', error)
  }
}

function createTimelineChart(data: Record<string, { date: string; value: number }[]>) {
  if (!data || Object.keys(data).length === 0) {
    timelineChart.value = {}
    return
  }

  timelineChart.value = useTimelineChart(data, 'line', {
    title: {
      text: 'Keyword Frequency Over Time',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'normal' },
    },
  })
}

async function exploreKeyword() {
  if (!selectedKeyword.value || !sourceStore.currentSource) {
    return
  }

  explorerLoading.value = true
  try {
    // Fetch documents containing this keyword
    const docsParams: any = { keyword: selectedKeyword.value, limit: 20 }
    if (selectedRunId.value) docsParams.run_id = selectedRunId.value

    const docsResponse = await api.get(
      `/keywords/${sourceStore.currentSource}/documents`,
      { params: docsParams }
    )
    keywordDocuments.value = docsResponse.data

    // Fetch co-occurring keywords
    const cooccurParams: any = { keyword: selectedKeyword.value, limit: 10 }
    if (selectedRunId.value) cooccurParams.run_id = selectedRunId.value

    const cooccurResponse = await api.get(
      `/keywords/${sourceStore.currentSource}/cooccurrence`,
      { params: cooccurParams }
    )
    keywordCoOccurrences.value = cooccurResponse.data
  } catch (error) {
    console.error('Failed to load keyword details:', error)
    keywordDocuments.value = []
    keywordCoOccurrences.value = []
  } finally {
    explorerLoading.value = false
  }
}

// Watch for filter changes
watch([searchQuery, minScore, selectedRunId], () => {
  currentPage.value = 1
  loadKeywords()
})

// Watch for chart count changes - only update charts from existing stats
watch([topKeywordsCount, wordcloudKeywordCount], () => {
  if (backendStats.value) {
    // Regenerate charts with new counts from existing stats
    if (backendStats.value.top_keywords) {
      topKeywordsChart.value = createTopKeywordsChart(
        backendStats.value.top_keywords,
        topKeywordsCount.value
      )
      const wordcloudData = createWordcloudData(
        backendStats.value.top_keywords,
        wordcloudKeywordCount.value
      )
      wordcloudChart.value = useWordCloud(wordcloudData, {
        title: {
          text: `Keyword Word Cloud (${wordcloudKeywordCount.value} Words)`,
          left: 'center',
          textStyle: { fontSize: 16, fontWeight: 'normal' },
        },
      })
    }
  }
})

// Watch for page changes
watch(currentPage, () => {
  loadKeywords()
})

function openDocumentInPageViewer(doc: KeywordDocument) {
  // Parse the page_id to extract issue_id and page number
  // Format: {source_id}_{YYYY-MM-DD}_{year_vol}_{issue_num}_{page_num}
  // Example: 3074409-X_1902-04-15_174_3_005
  const pageId = doc.page_id || doc.doc_id
  const parts = pageId.split('_')

  if (parts.length >= 5) {
    // Issue ID is first 4 parts (without page number)
    const issueId = parts.slice(0, 4).join('_')
    const pageNumber = parseInt(parts[4])

    // Open the issue view with the specific page number as query param
    const url = `/issue/${issueId}?page=${pageNumber}`
    window.open(url, '_blank')
  } else {
    console.warn('Invalid page ID format:', pageId)
  }
}

function onMetadataLoaded() {
  loadKeywords()
  loadTimeline()
}

onMounted(() => {
  if (sourceStore.currentSource) {
    // Metadata will trigger load via onMetadataLoaded
  }
})
</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex flex-wrap items-start gap-4">
        <!-- Title -->
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Keywords"
            description="Explore keyword extraction and relevance analysis."
            icon="keywords"
          />
        </div>

        <!-- Results Selector -->
        <div class="flex-1 min-w-[200px] self-stretch">
          <ResultsViewer
            v-if="sourceStore.currentSource"
            ref="resultsViewer"
            :source="sourceStore.currentSource"
            analysis-type="keywords"
            v-model:run-id="selectedRunId"
            @loaded="onMetadataLoaded"
          />
        </div>

        <!-- Inline Filters Card -->
        <div class="rounded-lg border bg-card p-3 flex items-center self-stretch">
          <div class="flex flex-col gap-2 w-full">
            <!-- Search -->
            <input
              v-model="searchQuery"
              type="text"
              placeholder="Search keywords..."
              class="w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
            />

            <!-- Score and Count Row -->
            <div class="flex items-center justify-between gap-3">
              <!-- Score Filter -->
              <div class="flex items-center gap-2 flex-1">
                <label class="text-xs text-muted-foreground whitespace-nowrap">
                  Score: {{ minScore }}%
                </label>
                <input
                  v-model.number="minScore"
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  class="flex-1"
                />
              </div>

              <!-- Results count -->
              <span class="text-xs text-muted-foreground whitespace-nowrap">
                {{ totalItems.toLocaleString() }}/{{ (backendStats?.total || 0).toLocaleString() }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 pb-6 space-y-6">
      <!-- Statistics Cards -->
      <StatisticsCards v-if="statistics" :stats="statsCards" :columns="5" />

      <!-- Timeline Chart -->
      <div v-if="keywords.length > 0 && timelineChart.series" class="rounded-lg border bg-card p-4">
        <VChart :option="timelineChart" class="h-[400px]" autoresize />
      </div>

      <!-- Distribution Charts Row -->
      <div v-if="keywords.length > 0" class="grid gap-4 md:grid-cols-2">
        <!-- Score Distribution -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="scoreDistributionChart" class="h-[300px]" autoresize />
        </div>

        <!-- Keywords per Document Distribution -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="keywordsPerDocChart" class="h-[300px]" autoresize />
        </div>
      </div>

      <!-- Top Keywords Chart -->
      <div v-if="keywords.length > 0" class="rounded-lg border bg-card p-4">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <label class="text-sm text-muted-foreground">Top N: {{ topKeywordsCount }}</label>
            <input
              v-model.number="topKeywordsCount"
              type="range"
              min="10"
              max="50"
              step="5"
              class="w-32"
            />
          </div>
        </div>
        <VChart :option="topKeywordsChart" class="h-[400px]" autoresize />
      </div>

      <!-- Wordcloud -->
      <div v-if="keywords.length > 0" class="rounded-lg border bg-card p-4">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <label class="text-sm text-muted-foreground">Words: {{ wordcloudKeywordCount }}</label>
            <input
              v-model.number="wordcloudKeywordCount"
              type="range"
              min="20"
              max="200"
              step="10"
              class="w-32"
            />
          </div>
        </div>
        <VChart :option="wordcloudChart" class="h-[600px]" autoresize />
      </div>

      <!-- Keyword Explorer -->
      <div v-if="keywords.length > 0" class="rounded-lg border bg-card p-6">
        <h3 class="text-lg font-semibold mb-2">Keyword Explorer</h3>
        <p class="text-sm text-muted-foreground mb-4">
          Explore documents and relationships for specific keywords
        </p>

        <!-- Search Input -->
        <div class="flex gap-3 mb-4">
          <select
            v-model="selectedKeyword"
            class="flex-1 px-3 py-2 text-sm"
          >
            <option value="">Select a keyword to explore...</option>
            <option
              v-for="kw in (backendStats?.top_keywords || []).slice(0, 100)"
              :key="kw.keyword"
              :value="kw.keyword"
            >
              {{ kw.keyword }} ({{ kw.frequency }})
            </option>
          </select>
          <button
            @click="exploreKeyword"
            :disabled="!selectedKeyword || explorerLoading"
            class="px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {{ explorerLoading ? 'Loading...' : 'Explore' }}
          </button>
        </div>

        <!-- Results Grid -->
        <div v-if="keywordDocuments.length > 0 || keywordCoOccurrences.length > 0" class="grid gap-4 md:grid-cols-2">
          <!-- Documents Column -->
          <div class="rounded-lg border bg-muted/50 p-4">
            <h4 class="font-semibold mb-3">
              Documents with "{{ selectedKeyword }}"
            </h4>
            <div v-if="explorerLoading" class="text-center py-8">
              <p class="text-sm text-muted-foreground">Loading...</p>
            </div>
            <div v-else-if="keywordDocuments.length > 0" class="space-y-2 max-h-[400px] overflow-y-auto">
              <div
                v-for="doc in keywordDocuments"
                :key="doc.doc_id"
                @click="openDocumentInPageViewer(doc)"
                class="p-3 rounded bg-card hover:bg-accent transition-colors cursor-pointer"
              >
                <div class="flex items-center justify-between">
                  <div class="flex-1 min-w-0">
                    <p class="text-sm font-medium truncate">{{ doc.doc_id }}</p>
                    <p v-if="doc.date" class="text-xs text-muted-foreground">{{ doc.date }}</p>
                  </div>
                  <span class="ml-2 px-2 py-1 bg-primary/10 text-primary text-xs rounded">
                    {{ doc.score.toFixed(3) }}
                  </span>
                </div>
              </div>
            </div>
            <div v-else class="text-center py-8">
              <p class="text-sm text-muted-foreground">No documents found</p>
            </div>
          </div>

          <!-- Co-occurring Keywords Column -->
          <div class="rounded-lg border bg-muted/50 p-4">
            <h4 class="font-semibold mb-3">
              Co-occurring Keywords
            </h4>
            <div v-if="explorerLoading" class="text-center py-8">
              <p class="text-sm text-muted-foreground">Loading...</p>
            </div>
            <div v-else-if="keywordCoOccurrences.length > 0" class="space-y-2 max-h-[400px] overflow-y-auto">
              <div
                v-for="(cooc, idx) in keywordCoOccurrences"
                :key="cooc.keyword"
                class="flex items-center justify-between p-3 rounded bg-card hover:bg-accent transition-colors"
              >
                <div class="flex items-center gap-2">
                  <span class="text-xs text-muted-foreground">{{ idx + 1 }}.</span>
                  <span class="text-sm font-medium">{{ cooc.keyword }}</span>
                </div>
                <span class="px-2 py-1 bg-secondary/10 text-secondary text-xs rounded">
                  {{ cooc.count }}
                </span>
              </div>
            </div>
            <div v-else class="text-center py-8">
              <p class="text-sm text-muted-foreground">No co-occurring keywords found</p>
            </div>
          </div>
        </div>
      </div>

      <!-- Loading state -->
      <div v-if="loading && keywords.length === 0" class="text-center py-12">
        <p class="text-muted-foreground">Loading keywords...</p>
      </div>

      <!-- Keyword table -->
      <div v-else-if="keywords.length > 0 || loading" class="rounded-lg border bg-card p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold">Keyword Details</h3>
          <div class="text-sm text-muted-foreground">
            <template v-if="!loading">
              Showing {{ keywords.length }} of {{ totalItems.toLocaleString() }} keywords
            </template>
            <template v-else>
              <span class="animate-pulse">Loading...</span>
            </template>
          </div>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="border-b">
              <tr>
                <th class="text-left p-2">Keyword</th>
                <th class="text-right p-2">Frequency</th>
                <th class="text-right p-2">TF-IDF Score</th>
              </tr>
            </thead>
            <tbody>
              <!-- Loading skeleton -->
              <template v-if="loading">
                <tr
                  v-for="i in pageSize"
                  :key="`skeleton-${i}`"
                  class="border-b"
                >
                  <td class="p-2">
                    <div class="h-4 bg-muted rounded animate-pulse"></div>
                  </td>
                  <td class="p-2">
                    <div class="h-4 bg-muted rounded animate-pulse ml-auto w-16"></div>
                  </td>
                  <td class="p-2">
                    <div class="h-4 bg-muted rounded animate-pulse ml-auto w-12"></div>
                  </td>
                </tr>
              </template>
              <!-- Actual data -->
              <template v-else>
                <tr
                  v-for="keyword in keywords"
                  :key="keyword.keyword"
                  class="border-b hover:bg-accent transition-colors"
                >
                  <td class="p-2 font-medium">{{ keyword.keyword }}</td>
                  <td class="p-2 text-right">{{ keyword.frequency.toLocaleString() }}</td>
                  <td class="p-2 text-right">{{ keyword.tfidf_score.toFixed(3) }}</td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>

        <!-- Pagination -->
        <PaginationControls
          :current-page="currentPage"
          :total-pages="totalPages"
          :loading="loading"
          @update:current-page="goToPage"
        />
      </div>

      <!-- No data state -->
      <div v-else-if="!loading" class="text-center py-12 text-muted-foreground">
        <p>No keywords found</p>
      </div>
    </div>
  </div>
</template>
