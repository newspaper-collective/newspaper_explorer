<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, PieChart, ScatterChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import { X } from 'lucide-vue-next'
import api from '@/lib/api'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import ResultsViewer from '@/components/ResultsViewer.vue'
import LayoutImageCard from '@/components/LayoutImageCard.vue'
import OpenSeadragonViewer from '@/components/OpenSeadragonViewer.vue'
import LayoutThumbnailGalleryDialog from '@/components/LayoutThumbnailGalleryDialog.vue'
import PaginationControls from '@/components/PaginationControls.vue'
import { getDetectionColor } from '@/lib/imageAnnotation'
import { useLayoutStats } from '@/lib/composables/useLayoutStats'
import type { EChartsOption } from 'echarts'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  PieChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
])

interface PageData {
  page_id: string
  image_path: string
  detection_count: number
  metadata: {
    date?: string
    issue_number?: string
    daily_count?: string
    page_number?: string
  }
  detections: Array<{
    detection_id: string
    class_name: string
    confidence: number
    bbox: {
      x1: number
      y1: number
      x2: number
      y2: number
    }
    text_content?: string
  }>
}

const sourceStore = useSourceStore()
const loading = ref(false)
const isChangingRun = ref(false)
const selectedRunId = ref<string | null>(null)
const resultsViewer = ref<InstanceType<typeof ResultsViewer>>()
const pages = ref<PageData[]>([])
const classNames = ref<string[]>([])

// Composables for stats and charts
const {
  backendStats,
  statistics,
} = useLayoutStats()

// Charts
const timelineChart = ref<EChartsOption>({})
const pieChart = ref<EChartsOption>({})

// Pagination
const currentPage = ref(1)
const pageSize = ref(4) // 4 images per page for thumbnail view

// Filters
const selectedClasses = ref<string[]>([])
const minConfidence = ref(0)

// Dialog for detection details
const detailsDialog = ref(false)
const selectedPageDetections = ref<any[]>([])

// Dialog for image viewer
const imageViewerDialog = ref(false)
const selectedImagePath = ref<string>('')
const selectedPageMetadata = ref<any>(null)
const selectedImageDetections = ref<any[]>([])

// Thumbnail gallery dialog
const thumbnailGalleryDialog = ref(false)
const thumbnailCurrentPage = ref(1)
const thumbnailPageSize = ref(48) // Show 48 thumbnails in the gallery

function openThumbnailGallery() {
  thumbnailGalleryDialog.value = true
  thumbnailCurrentPage.value = 1
  loadThumbnailPages()
}

function closeThumbnailGallery() {
  thumbnailGalleryDialog.value = false
}

function selectThumbnail(page: PageData) {
  closeThumbnailGallery()
  viewImage(page.image_path, page.metadata, page.detections)
}

function changeThumbnailPage(page: number) {
  thumbnailCurrentPage.value = page
  loadThumbnailPages()
}

const thumbnailPages = ref<PageData[]>([])
const thumbnailTotalPages = computed(() => Math.ceil(totalPages.value / thumbnailPageSize.value))

async function loadThumbnailPages() {
  if (!sourceStore.currentSource || !selectedRunId.value) return

  loading.value = true
  try {
    const response = await api.get(
      `/layout/${sourceStore.currentSource}/pages`,
      {
        params: {
          run_id: selectedRunId.value,
          page: thumbnailCurrentPage.value,
          page_size: thumbnailPageSize.value,
          min_confidence: minConfidence.value / 100,
          classes: selectedClasses.value.length > 0 ? selectedClasses.value.join(',') : undefined,
        }
      }
    )
    thumbnailPages.value = response.data.pages || []
  } catch (error) {
    console.error('Failed to load thumbnail pages:', error)
  } finally {
    loading.value = false
  }
}

// Enhanced stats from detailed backend call
const enhancedStats = computed(() => {
  if (!backendStats.value) return null

  const stats = backendStats.value
  const avgDetectionsPerPage = stats.unique_pages && stats.total
    ? (stats.total / stats.unique_pages).toFixed(1)
    : '0'

  // Calculate text vs image ratio from counts
  const textClasses = ['Paragraph', 'Heading', 'Caption', 'Text']
  const imageClasses = ['Image', 'Picture', 'Photo', 'Illustration']

  let textCount = 0
  let imageCount = 0
  let totalOtherCount = 0

  // Access counts from the stats object (it's in the Any type from backend)
  const counts = (stats as any).counts
  if (counts) {
    Object.entries(counts).forEach(([className, count]) => {
      if (textClasses.some(tc => className.toLowerCase().includes(tc.toLowerCase()))) {
        textCount += count as number
      } else if (imageClasses.some(ic => className.toLowerCase().includes(ic.toLowerCase()))) {
        imageCount += count as number
      } else {
        totalOtherCount += count as number
      }
    })
  }

  const totalTextImage = textCount + imageCount
  const textPercentage = totalTextImage > 0 ? ((textCount / totalTextImage) * 100).toFixed(1) : '0'
  const imagePercentage = totalTextImage > 0 ? ((imageCount / totalTextImage) * 100).toFixed(1) : '0'

  return {
    totalDetections: stats.total || 0,
    uniquePages: stats.unique_pages || 0,
    uniqueClasses: stats.unique_classes || 0,
    avgDetectionsPerPage,
    avgConfidence: stats.avg_confidence ? (stats.avg_confidence * 100).toFixed(1) : '0',
    avgWidth: stats.avg_width ? Math.round(stats.avg_width) : 0,
    avgHeight: stats.avg_height ? Math.round(stats.avg_height) : 0,
    textCount,
    imageCount,
    otherCount: totalOtherCount,
    textPercentage,
    imagePercentage,
  }
})

const hasData = computed(() => pages.value.length > 0)

async function loadClassNames() {
  if (!sourceStore.currentSource || !selectedRunId.value) return

  try {
    console.log('🏷️  Loading class names for', selectedRunId.value)
    const response = await api.get(
      `/layout/${sourceStore.currentSource}/labels`,
      { params: { run_id: selectedRunId.value } }
    )
    classNames.value = response.data
    selectedClasses.value = [...response.data]
    console.log('✅ Loaded', response.data.length, 'classes:', response.data)
  } catch (error) {
    console.error('Failed to load class names:', error)
  }
}

async function loadStats() {
  if (!sourceStore.currentSource || !selectedRunId.value) return

  try {
    console.log('📊 Loading stats for', selectedRunId.value)

    // Load unfiltered stats for overview
    const unfilteredResponse = await api.get(
      `/layout/${sourceStore.currentSource}/stats`,
      { params: { run_id: selectedRunId.value, min_confidence: 0 } }
    )
    backendStats.value = unfilteredResponse.data

    // Load filtered stats for pie chart
    const filteredResponse = await api.get(
      `/layout/${sourceStore.currentSource}/stats`,
      { params: { run_id: selectedRunId.value, min_confidence: minConfidence.value / 100 } }
    )

    // Generate pie chart from filtered stats
    if (filteredResponse.data.counts) {
      pieChart.value = createPieChartFromCounts(filteredResponse.data.counts) as EChartsOption
    }

    console.log('✅ Stats loaded')
  } catch (error) {
    console.error('Failed to load stats:', error)
  }
}

async function loadTimeline() {
  if (!sourceStore.currentSource || !selectedRunId.value) return

  try {
    const response = await api.get(
      `/layout/${sourceStore.currentSource}/timeline`,
      { params: { aggregation: 'day', run_id: selectedRunId.value } }
    )
    console.log('📊 Timeline data loaded')

    // Create timeline chart with all classes stacked
    timelineChart.value = createMultiClassTimelineChart(response.data)
  } catch (error) {
    console.error('Failed to load timeline:', error)
  }
}

function createMultiClassTimelineChart(data: Record<string, { date: string; value: number }[]>) {
  if (!data || Object.keys(data).length === 0) {
    return {}
  }

  // Get all unique dates
  const allDates = new Set<string>()
  Object.values(data).forEach(series => {
    series.forEach(point => allDates.add(point.date))
  })
  const sortedDates = Array.from(allDates).sort()

  // Create series for each class
  const series = Object.entries(data).map(([className, points]) => {
    // Create a map for quick lookup
    const valueMap = new Map(points.map(p => [p.date, p.value]))

    // Fill in data for all dates (0 if no data)
    const seriesData = sortedDates.map(date => valueMap.get(date) || 0)

    return {
      name: className,
      type: 'line' as const,
      stack: 'total',
      areaStyle: {},
      emphasis: {
        focus: 'series' as const
      },
      data: seriesData,
      itemStyle: {
        color: getDetectionColor(className)
      },
      lineStyle: {
        color: getDetectionColor(className)
      }
    }
  })

  return {
    title: {
      text: 'Layout Detections Over Time',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'normal' as const }
    },
    tooltip: {
      trigger: 'axis' as const,
      axisPointer: {
        type: 'cross' as const,
        label: {
          backgroundColor: '#6a7985'
        }
      }
    },
    legend: {
      data: Object.keys(data),
      top: 30,
      textStyle: { fontSize: 11 }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: 80,
      containLabel: true
    },
    xAxis: {
      type: 'category' as const,
      boundaryGap: false,
      data: sortedDates
    },
    yAxis: {
      type: 'value' as const
    },
    series
  }
}

function createPieChartFromCounts(counts: Record<string, number>) {
  if (!counts || Object.keys(counts).length === 0) {
    return {}
  }

  const sortedEntries = Object.entries(counts).sort((a, b) => b[1] - a[1])
  const data = sortedEntries.map(([name, value]) => ({
    name,
    value,
    itemStyle: { color: getDetectionColor(name) }
  }))

  return {
    title: {
      text: 'Detection Class Distribution',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'normal' as const },
    },
    tooltip: {
      trigger: 'item' as const,
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical' as const,
      left: 'left',
      top: '15%',
      textStyle: { fontSize: 12 }
    },
    series: [
      {
        type: 'pie' as const,
        radius: ['40%', '70%'],
        center: ['60%', '55%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 4,
          borderColor: '#fff',
          borderWidth: 2
        },
        label: {
          show: true,
          formatter: '{d}%',
          fontSize: 11
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 14,
            fontWeight: 'bold' as const
          }
        },
        data
      }
    ]
  }
}

const totalPages = ref(0)

async function loadPages() {
  if (!sourceStore.currentSource || !selectedRunId.value) {
    console.log('⚠️ loadPages called but missing source or runId')
    return
  }

  loading.value = true
  pages.value = []

  try {
    const params: any = {
      page: currentPage.value,
      page_size: pageSize.value,
      run_id: selectedRunId.value,
    }

    if (selectedClasses.value.length < classNames.value.length) {
      params.class_filter = selectedClasses.value
    }
    if (minConfidence.value > 0) params.min_confidence = minConfidence.value / 100

    const response = await api.get(
      `/layout/${sourceStore.currentSource}/pages`,
      { params }
    )

    pages.value = response.data.pages
    totalPages.value = response.data.total
  } catch (error) {
    console.error('❌ Failed to load pages:', error)
    pages.value = []
    totalPages.value = 0
  } finally {
    loading.value = false
  }
}

async function applyFilters() {
  currentPage.value = 1
  await loadPages()
}

function goToPage(page: number) {
  currentPage.value = page
  loadPages()
}

function showDetails(detections: any[]) {
  selectedPageDetections.value = detections
  detailsDialog.value = true
}

function viewImage(imagePath: string, metadata: any, detections: any[]) {
  selectedImagePath.value = imagePath
  selectedPageMetadata.value = metadata
  selectedImageDetections.value = detections
  imageViewerDialog.value = true
}

const totalPagesCount = computed(() => {
  return Math.ceil(totalPages.value / pageSize.value)
})

// Watch for filter changes
watch([selectedClasses, minConfidence], () => {
  console.log('🔍 Filter changed')
  if (selectedRunId.value && !isChangingRun.value) {
    applyFilters()
  }
}, { deep: true })

// Watch for run changes
watch(selectedRunId, async (newId, oldId) => {
  console.log('🔄 selectedRunId changed from', oldId, 'to', newId)
  if (!newId) {
    pages.value = []
    classNames.value = []
    selectedClasses.value = []
    timelineChart.value = {}
    pieChart.value = {}
    return
  }

  if (newId !== oldId) {
    isChangingRun.value = true

    try {
      pages.value = []
      classNames.value = []
      selectedClasses.value = []
      currentPage.value = 1
      totalPages.value = 0
      timelineChart.value = {}
      pieChart.value = {}

      await loadClassNames()
      await loadPages()
      await loadStats()
      await loadTimeline()
    } catch (error) {
      console.error('❌ Error loading run data:', error)
    } finally {
      isChangingRun.value = false
    }
  }
}, { immediate: false })

function onMetadataLoaded() {
  console.log('📥 onMetadataLoaded called')
}

onMounted(() => {
  // Data loaded when ResultsViewer emits @loaded
})

watch(() => sourceStore.currentSource, () => {
  selectedRunId.value = null
  pages.value = []
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
            title="Layout Analysis"
            description="Explore detected layout regions in newspaper pages"
            icon="layout"
          />
        </div>

        <!-- Results Selector -->
        <div class="flex-1 min-w-[200px] self-stretch">
          <ResultsViewer
            v-if="sourceStore.currentSource"
            ref="resultsViewer"
            :source="sourceStore.currentSource"
            analysis-type="layout"
            v-model:run-id="selectedRunId"
            @loaded="onMetadataLoaded"
          />
        </div>

        <!-- Inline Filters Card -->
        <div class="rounded-lg border bg-card p-3 flex items-center self-stretch">
          <div class="flex flex-col gap-2 w-full min-w-[250px] max-w-[600px]">
            <!-- Class Filter Checkboxes -->
            <div v-if="classNames.length > 0" class="flex flex-wrap gap-1.5 max-w-full">
              <label
                v-for="className in classNames"
                :key="className"
                class="inline-flex items-center gap-1.5 px-2 py-1 border rounded-md cursor-pointer hover:bg-accent transition-colors text-xs whitespace-nowrap"
                :class="{ 'bg-accent': selectedClasses.includes(className) }"
              >
                <input
                  type="checkbox"
                  :value="className"
                  v-model="selectedClasses"
                  class="rounded w-3 h-3"
                />
                <span
                  class="w-2.5 h-2.5 rounded-sm border border-black"
                  :style="{ backgroundColor: getDetectionColor(className) }"
                />
                <span>{{ className }}</span>
              </label>
            </div>

            <!-- Confidence and Count Row -->
            <div class="flex items-center justify-between gap-3">
              <div class="flex items-center gap-2 flex-1">
                <label class="text-xs text-muted-foreground whitespace-nowrap">
                  Conf: {{ minConfidence }}%
                </label>
                <input
                  v-model.number="minConfidence"
                  type="range"
                  min="0"
                  max="100"
                  step="5"
                  class="flex-1"
                />
              </div>

              <span v-if="statistics" class="text-xs text-muted-foreground whitespace-nowrap">
                {{ statistics.uniquePages }} pages
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 pb-6 space-y-6">
      <!-- Statistics and Charts Section (Collapsible) -->
      <details v-if="hasData || loading" class="rounded-lg border bg-card" open>
        <summary class="cursor-pointer p-4 hover:bg-accent/50 transition-colors font-semibold select-none">
          📊 Statistics & Charts
        </summary>
        <div class="p-6 pt-2 space-y-6">
          <!-- Statistics Cards - 2 rows of 4 cards -->
          <div v-if="enhancedStats" class="grid grid-cols-2 md:grid-cols-4 gap-4">
            <!-- Row 1 -->
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.totalDetections.toLocaleString() }}</div>
              <p class="text-xs text-muted-foreground">Total Detections</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.uniquePages.toLocaleString() }}</div>
              <p class="text-xs text-muted-foreground">Unique Pages</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.textPercentage }}% / {{ enhancedStats.imagePercentage }}%</div>
              <p class="text-xs text-muted-foreground">Text / Image Ratio</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.avgDetectionsPerPage }}</div>
              <p class="text-xs text-muted-foreground">Avg per Page</p>
            </div>

            <!-- Row 2 -->
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.avgConfidence }}%</div>
              <p class="text-xs text-muted-foreground">Avg Confidence</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.textCount.toLocaleString() }}</div>
              <p class="text-xs text-muted-foreground">Text Regions ({{ enhancedStats.textPercentage }}%)</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.imageCount.toLocaleString() }}</div>
              <p class="text-xs text-muted-foreground">Image Regions ({{ enhancedStats.imagePercentage }}%)</p>
            </div>
            <div class="rounded-lg border bg-card p-4">
              <div class="text-2xl font-bold">{{ enhancedStats.otherCount.toLocaleString() }}</div>
              <p class="text-xs text-muted-foreground">Other Regions</p>
            </div>
          </div>

          <!-- Charts Grid -->
          <div class="grid gap-4 md:grid-cols-2">
            <!-- Timeline Chart -->
            <div v-if="timelineChart.series" class="rounded-lg border bg-card p-4">
              <VChart :option="timelineChart" class="h-[400px]" autoresize />
            </div>

            <!-- Pie Chart -->
            <div v-if="pieChart.series" class="rounded-lg border bg-card p-4">
              <VChart :option="pieChart" class="h-[400px]" autoresize />
            </div>
          </div>
        </div>
      </details>

      <!-- Loading state (only for initial load) -->
      <div v-if="loading && totalPages === 0" class="text-center py-12">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
        <p class="text-muted-foreground">Loading layout detections...</p>
      </div>

      <!-- Page Gallery -->
      <div v-else-if="totalPages > 0 || loading" class="rounded-lg border bg-card p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold">
            Page Gallery ({{ totalPages.toLocaleString() }} pages)
          </h3>
          <div class="flex items-center gap-4">
            <button
              @click="openThumbnailGallery"
              class="p-2 rounded-lg border border-border hover:bg-accent transition-colors"
              title="View all thumbnails"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM14 5a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1V5zM4 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1H5a1 1 0 01-1-1v-4zM14 15a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
              </svg>
            </button>
            <div class="text-sm text-muted-foreground">
              Page {{ currentPage }} of {{ totalPagesCount }}
            </div>
          </div>
        </div>

        <!-- Page Grid - 4x1 layout -->
        <div class="grid gap-6 grid-cols-4">
          <template v-if="loading">
            <!-- Loading placeholders matching LayoutImageCard structure -->
            <div
              v-for="i in pageSize"
              :key="`placeholder-${i}`"
              class="rounded-lg border bg-card overflow-hidden"
            >
              <!-- Header skeleton -->
              <div class="p-4 border-b bg-muted/50 space-y-2">
                <div class="h-5 bg-muted rounded animate-pulse w-3/4"></div>
                <div class="h-3 bg-muted rounded animate-pulse w-1/2"></div>
                <div class="h-4 bg-muted rounded animate-pulse w-2/5"></div>
              </div>

              <!-- Image skeleton with black background -->
              <div class="bg-black" style="height: 384px">
                <div class="w-full h-full bg-muted/20 animate-pulse"></div>
              </div>

              <!-- Footer skeleton -->
              <div class="p-3 border-t flex justify-end">
                <div class="h-8 bg-muted rounded animate-pulse w-28"></div>
              </div>
            </div>
          </template>
          <template v-else>
            <LayoutImageCard
              v-for="(page, index) in pages"
              :key="`${selectedRunId}-${currentPage}-${index}-${page.page_id}`"
              :page-id="page.page_id"
              :image-path="page.image_path"
              :detections="page.detections"
              :metadata="page.metadata"
              @view-details="showDetails(page.detections)"
              @view-image="viewImage(page.image_path, page.metadata, page.detections)"
            />
          </template>
        </div>

        <!-- Pagination -->
        <PaginationControls
          :current-page="currentPage"
          :total-pages="totalPagesCount"
          :loading="loading"
          @update:current-page="goToPage"
        />
      </div>

      <!-- No data message -->
      <div v-else class="rounded-lg border bg-card p-12 text-center">
        <p class="text-lg font-semibold mb-2">No layout detections found</p>
        <p class="text-muted-foreground">
          Try adjusting the filters or run layout detection first.
        </p>
      </div>
    </div>

    <!-- Detection details dialog -->
    <Teleport to="body">
      <div
        v-if="detailsDialog"
        class="fixed inset-0 z-[100] flex items-center justify-center bg-black/50"
        @click="detailsDialog = false"
      >
      <div
        class="bg-card rounded-lg shadow-lg max-w-4xl w-full max-h-[80vh] overflow-hidden m-4"
        @click.stop
      >
        <div class="flex items-center justify-between p-6 border-b">
          <h2 class="text-lg font-semibold">Detection Details</h2>
          <button
            @click="detailsDialog = false"
            class="p-2 hover:bg-accent rounded-md transition-colors"
          >
            <X class="w-5 h-5" />
          </button>
        </div>

        <div class="p-6 overflow-y-auto max-h-[60vh]">
          <div class="flex flex-wrap gap-3 justify-center">
            <div
              v-for="detection in selectedPageDetections"
              :key="detection.detection_id"
              class="inline-flex items-center gap-2 px-3 py-2 border rounded-md bg-card"
            >
              <span
                class="w-4 h-4 rounded-sm border-2 border-black flex-shrink-0"
                :style="{ backgroundColor: getDetectionColor(detection.class_name) }"
              />
              <span class="text-sm font-medium">
                {{ detection.class_name }} {{ detection.confidence.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>

        <div class="p-4 border-t flex justify-end">
          <button
            @click="detailsDialog = false"
            class="px-4 py-2 border rounded-md hover:bg-accent transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </Teleport>

    <!-- Image viewer dialog -->
    <Teleport to="body">
      <div
        v-if="imageViewerDialog"
        class="fixed inset-0 z-[100] flex items-center justify-center bg-black/90"
        @click="imageViewerDialog = false"
      >
      <div
        class="bg-card rounded-lg shadow-lg w-[90vw] max-w-[1400px] h-[95vh] overflow-hidden m-4 flex flex-col"
        @click.stop
      >
        <div class="flex items-center justify-between p-4 border-b bg-muted/50">
          <div v-if="selectedPageMetadata" class="space-y-1">
            <p class="font-semibold text-base">
              {{ selectedPageMetadata.date }}
            </p>
            <p class="text-xs text-muted-foreground">
              Issue {{ selectedPageMetadata.issue_number }} • Daily {{ selectedPageMetadata.daily_count }} • Page {{ selectedPageMetadata.page_number }}
            </p>
          </div>
          <h2 v-else class="text-lg font-semibold">Image Viewer</h2>
          <button
            @click="imageViewerDialog = false"
            class="p-2 hover:bg-accent rounded-md transition-colors"
          >
            <X class="w-5 h-5" />
          </button>
        </div>

        <div class="flex-1 relative">
          <OpenSeadragonViewer
            v-if="selectedImagePath && sourceStore.currentSource"
            :image-url="`/static/${sourceStore.currentSource}/images/${selectedImagePath}`"
            :current-page="1"
            :total-pages="1"
            :detections="selectedImageDetections"
          />
        </div>
      </div>
    </Teleport>

    <!-- Thumbnail Gallery Dialog -->
    <LayoutThumbnailGalleryDialog
      v-if="thumbnailGalleryDialog"
      :open="thumbnailGalleryDialog"
      :pages="thumbnailPages"
      :current-page="thumbnailCurrentPage"
      :total-pages="thumbnailTotalPages"
      :total-items="totalPages"
      :loading="loading"
      :source="sourceStore.currentSource || ''"
      @close="closeThumbnailGallery"
      @select="selectThumbnail"
      @page-change="changeThumbnailPage"
    />
  </div>
</template>
