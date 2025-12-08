<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, PieChart, LineChart, ScatterChart } from 'echarts/charts'
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
import FilterBar from '@/components/FilterBar.vue'
import PictureCard from '@/components/PictureCard.vue'
import PictureDetailDialog from '@/components/PictureDetailDialog.vue'
import ThumbnailGalleryDialog from '@/components/ThumbnailGalleryDialog.vue'
import PaginationControls from '@/components/PaginationControls.vue'
import type { EChartsOption } from 'echarts'

// Composables
import {
  formatDate,
  parsePageMetadata,
  getFullImageUrl,
  extractDateFromPageId,
} from '@/lib/composables/useImageUtils'
import { useImageCropping } from '@/lib/composables/useImageCropping'
import { usePagination } from '@/lib/composables/usePagination'
import { useLayoutStats } from '@/lib/composables/useLayoutStats'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  PieChart,
  LineChart,
  ScatterChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
])

interface Picture {
  detection_id: string
  page_id: string
  class_name: string
  confidence: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  date?: string
  image_path?: string
  newspaper_title?: string
  text_content?: string  // Caption text from enriched data (deprecated - use caption_bbox)
  caption_id?: string    // ID of nearest caption detection
  caption_bbox?: {       // Bounding box of nearest caption for cropping
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

const sourceStore = useSourceStore()
const pictures = ref<Picture[]>([])
const loading = ref(false)
const selectedRunId = ref<string | null>(null)
const resultsViewer = ref<InstanceType<typeof ResultsViewer>>()
const searchQuery = ref('')
const minConfidence = ref(25)
const excludeHeadersFooters = ref(true)
const headerFooterThreshold = ref(10)
const minHeight = ref(0)
const onlyWithCaptions = ref(false)
const errorMessage = ref<string | null>(null)

// Filter options for FilterBar component
const filterOptions = computed(() => ({
  searchQuery: searchQuery.value,
  minConfidence: minConfidence.value,
  onlyWithCaptions: onlyWithCaptions.value,
  excludeHeadersFooters: excludeHeadersFooters.value,
  headerFooterThreshold: headerFooterThreshold.value,
  minHeight: minHeight.value,
}))

function updateFilterOptions(newOptions: typeof filterOptions.value) {
  searchQuery.value = newOptions.searchQuery
  minConfidence.value = newOptions.minConfidence
  onlyWithCaptions.value = newOptions.onlyWithCaptions
  excludeHeadersFooters.value = newOptions.excludeHeadersFooters
  headerFooterThreshold.value = newOptions.headerFooterThreshold
  minHeight.value = newOptions.minHeight
}

// Statistics cards data
const statsCards = computed(() => {
  if (!statistics.value) return []
  return [
    { label: 'Total Pictures', value: statistics.value.totalPictures, format: 'number' as const },
    { label: 'Unique Pages', value: statistics.value.uniquePages, format: 'number' as const },
    { label: 'Avg Confidence', value: statistics.value.avgConfidence, format: 'percentage' as const },
    { label: 'Avg Width', value: statistics.value.avgWidth, format: 'pixels' as const },
    { label: 'Avg Height', value: statistics.value.avgHeight, format: 'pixels' as const },
  ]
})

// Composables
const { 
  currentPage, 
  totalItems, 
  totalPages,
  pageSize,
  goToPage 
} = usePagination({ itemsPerPage: 4 })

const {
  croppedImages,
  croppedCaptions,
  cropLoadingImages,
  cropLoadingCaptions,
  loadCroppedImage: cropImage,
  loadCroppedCaption: cropCaption,
  clearCache: clearImageCache,
} = useImageCropping()

const {
  backendStats,
  statistics,
  createTimelineChart,
  createConfidenceChart,
  createPositionChart,
  createSizeChart,
} = useLayoutStats()

// Dialog for image viewer
const imageViewerDialog = ref(false)
const selectedImagePath = ref<string>('')
const selectedPageMetadata = ref<any>(null)
const selectedImageDetections = ref<any[]>([])
const selectedPictureForDialog = ref<Picture | null>(null)
const allPageDetections = ref<any[]>([])

// Thumbnail gallery dialog
const thumbnailGalleryOpen = ref(false)
const {
  currentPage: galleryPage,
  totalItems: galleryTotalItems,
  totalPages: galleryTotalPages,
  pageSize: galleryPageSize,
  goToPage: goToGalleryPage
} = usePagination({ itemsPerPage: 48 })
const galleryPictures = ref<Picture[]>([])
const galleryLoading = ref(false)

// Chart options (generated from stats)
const confidenceDistributionChart = ref<EChartsOption>({})
const timelineChart = ref<EChartsOption>({})
const positionDistributionChart = ref<EChartsOption>({})
const sizeDistributionChart = ref<EChartsOption>({})

// Watch for page changes to reload data
watch(currentPage, () => {
  loadPictures()
})

// Watch for gallery page changes
watch(galleryPage, () => {
  if (thumbnailGalleryOpen.value) {
    loadGalleryPictures()
  }
})

// Watch for filter changes to reload data (reset to page 1)
watch([minConfidence, selectedRunId, excludeHeadersFooters, onlyWithCaptions, minHeight, searchQuery], () => {
  currentPage.value = 1
  loadPictures()
})



async function loadGalleryPictures() {
  if (!sourceStore.currentSource) return

  galleryLoading.value = true
  try {
    const params: any = {
      label: 'Picture',
      min_confidence: minConfidence.value / 100,
      page: galleryPage.value,
      page_size: galleryPageSize.value,
      exclude_headers_footers: excludeHeadersFooters.value,
      header_footer_threshold: headerFooterThreshold.value,
    }
    if (selectedRunId.value) params.run_id = selectedRunId.value
    if (onlyWithCaptions.value) params.only_with_captions = true
    if (minHeight.value > 0) params.min_height = minHeight.value
    if (searchQuery.value.trim()) params.search = searchQuery.value.trim()

    const response = await api.get(
      `/layout/${sourceStore.currentSource}/detections`,
      { params }
    )

    const allPictures: Picture[] = response.data.items.map((detection: any) => {
      let date = detection.date
      if (!date) {
        const dateFromId = extractDateFromPageId(detection.page_id)
        if (dateFromId) date = dateFromId
      }

      return {
        detection_id: detection.detection_id,
        page_id: detection.page_id,
        image_path: detection.image_path,
        bbox_x1: detection.bbox_x1,
        bbox_y1: detection.bbox_y1,
        bbox_x2: detection.bbox_x2,
        bbox_y2: detection.bbox_y2,
        confidence: detection.confidence,
        date: date || undefined,
        caption_text: detection.caption_text,
        caption_bbox: detection.caption_bbox,
        class_name: 'Picture',
      }
    })

    galleryPictures.value = allPictures
    galleryTotalItems.value = response.data.total

    // Images will be loaded lazily by the ThumbnailGalleryDialog intersection observer
  } catch (error) {
    console.error('Failed to load gallery pictures:', error)
  } finally {
    galleryLoading.value = false
  }
}

function openThumbnailGallery() {
  thumbnailGalleryOpen.value = true
  // Always load fresh data when opening
  loadGalleryPictures()
}

function selectFromGallery(picture: Picture) {
  thumbnailGalleryOpen.value = false
  // Ensure the image is cropped before viewing
  loadPictureCrop(picture)
  if (picture.caption_bbox) {
    loadCaptionCrop(picture)
  }
  // Call viewImage to properly load all page detections
  viewImage(picture)
}

async function loadPictures() {
  if (!sourceStore.currentSource) return

  loading.value = true
  errorMessage.value = null  // Clear any previous errors
  try {
    // 1a. Load UNFILTERED statistics for the statistics cards (total dataset overview)
    const unfilteredStatsParams: any = {
      label: 'Picture',
      min_confidence: 0, // No filter - show all pictures
    }
    if (selectedRunId.value) unfilteredStatsParams.run_id = selectedRunId.value
    
    const unfilteredStatsResponse = await api.get(
      `/layout/${sourceStore.currentSource}/stats`,
      { params: unfilteredStatsParams }
    )
    
    // Store unfiltered stats for statistics cards (total dataset)
    backendStats.value = unfilteredStatsResponse.data
    
    // 1b. Load FILTERED statistics for charts (respects current filters)
    const statsParams: any = {
      label: 'Picture',
      min_confidence: minConfidence.value / 100,
    }
    if (selectedRunId.value) statsParams.run_id = selectedRunId.value
    
    const statsResponse = await api.get(
      `/layout/${sourceStore.currentSource}/stats`,
      { params: statsParams }
    )
    
    updateChartsFromStats(statsResponse.data)

    // 2. Load paginated pictures with all filters
    const params: any = {
      label: 'Picture',
      min_confidence: minConfidence.value / 100,
      include_captions: true,
      page: currentPage.value,
      page_size: pageSize.value,
      exclude_headers_footers: excludeHeadersFooters.value,
      header_footer_threshold: headerFooterThreshold.value,
      only_with_captions: onlyWithCaptions.value,
      min_height: minHeight.value,
    }
    if (selectedRunId.value) params.run_id = selectedRunId.value
    if (searchQuery.value.trim()) params.search = searchQuery.value.trim()

    const response = await api.get(
      `/layout/${sourceStore.currentSource}/detections`,
      { params }
    )
    
    // Transform detections into Picture objects
    const allPictures: Picture[] = response.data.items.map((detection: any) => {
      // Extract date from page_id format: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}
      let date = detection.date
      if (!date) {
        const dateMatch = detection.page_id?.match(/_(\d{4}-\d{2}-\d{2})_/)
        if (dateMatch) {
          date = dateMatch[1]
        }
      }

      return {
        detection_id: detection.detection_id,
        page_id: detection.page_id,
        class_name: detection.class_name,
        confidence: detection.confidence,
        bbox_x1: detection.bbox?.x1 ?? 0,
        bbox_y1: detection.bbox?.y1 ?? 0,
        bbox_x2: detection.bbox?.x2 ?? 0,
        bbox_y2: detection.bbox?.y2 ?? 0,
        date,
        image_path: detection.image_path,
        newspaper_title: detection.newspaper_title,
        text_content: detection.text_content,
        caption_id: detection.caption_id,
        caption_bbox: detection.caption_bbox,
      }
    })
    
    pictures.value = allPictures
    // Update total pages based on backend response
    // We need to store total count somewhere to calculate total pages
    // Let's add a ref for totalItems
    totalItems.value = response.data.total
    
  } catch (error: any) {
    console.error('Failed to load pictures:', error)
    
    // Check if this is a caption enrichment error
    if (error.response?.status === 400 && error.response?.data?.detail?.includes('Caption data not available')) {
      errorMessage.value = error.response.data.detail
      // Automatically uncheck the filter to let user continue
      onlyWithCaptions.value = false
    } else {
      errorMessage.value = 'Failed to load pictures. Please try again.'
    }
    
    pictures.value = []
    totalItems.value = 0
  } finally {
    loading.value = false
  }
}

function updateChartsFromStats(stats: any) {
  // Use composable chart generators
  if (stats.timeline) {
    timelineChart.value = createTimelineChart(stats.timeline)
  }
  
  if (stats.confidence_distribution) {
    confidenceDistributionChart.value = createConfidenceChart(stats.confidence_distribution)
  }

  if (stats.position_distribution) {
    positionDistributionChart.value = createPositionChart(stats.position_distribution)
  }

  if (stats.size_distribution) {
    sizeDistributionChart.value = createSizeChart(stats.size_distribution)
  }
}



// Wrapper functions for Picture-specific signatures
function getImageUrl(picture: Picture): string | null {
  if (!picture.image_path || !sourceStore.currentSource) return null
  return getFullImageUrl(picture.image_path, sourceStore.currentSource)
}

async function loadPictureCrop(picture: Picture) {
  const cacheKey = picture.detection_id
  const imageUrl = getImageUrl(picture)
  if (!imageUrl) return
  
  const bbox = {
    x1: picture.bbox_x1,
    y1: picture.bbox_y1,
    x2: picture.bbox_x2,
    y2: picture.bbox_y2,
  }
  
  await cropImage(imageUrl, bbox, cacheKey)
}

async function loadCaptionCrop(picture: Picture) {
  if (!picture.caption_bbox) return
  
  const cacheKey = `caption_${picture.detection_id}`
  const imageUrl = getImageUrl(picture)
  if (!imageUrl) return
  
  await cropCaption(imageUrl, picture.caption_bbox, cacheKey)
}

// Load cropped images for current page pictures
watch(pictures, async (currentPictures) => {
  for (const picture of currentPictures) {
    await loadPictureCrop(picture)
    if (picture.caption_bbox) {
      await loadCaptionCrop(picture)
    }
  }
}, { immediate: true })

// Page thumbnail with detection overlay
const pageThumbnail = ref<string>('')
const pageThumbnailLoading = ref(false)

async function loadPageThumbnailWithOverlay(picture: Picture) {
  if (pageThumbnailLoading.value) return
  
  pageThumbnailLoading.value = true
  pageThumbnail.value = ''
  
  const imageUrl = getImageUrl(picture)
  if (!imageUrl) {
    pageThumbnailLoading.value = false
    return
  }

  try {
    // Load the full image
    const img = new Image()
    img.crossOrigin = 'anonymous'
    
    await new Promise<void>((resolve, reject) => {
      img.onload = () => resolve()
      img.onerror = () => reject(new Error('Failed to load image'))
      img.src = imageUrl
    })

    // Create canvas for thumbnail
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      pageThumbnailLoading.value = false
      return
    }

    // Scale to thumbnail size (max 300px width)
    const maxWidth = 300
    const scale = maxWidth / img.width
    canvas.width = maxWidth
    canvas.height = img.height * scale

    // Draw the full page
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    // Draw picture detection box
    ctx.strokeStyle = '#3b82f6' // blue
    ctx.lineWidth = 2
    ctx.strokeRect(
      picture.bbox_x1 * scale,
      picture.bbox_y1 * scale,
      (picture.bbox_x2 - picture.bbox_x1) * scale,
      (picture.bbox_y2 - picture.bbox_y1) * scale
    )

    // Draw caption detection box if available
    if (picture.caption_bbox) {
      ctx.strokeStyle = '#10b981' // green
      ctx.lineWidth = 2
      ctx.strokeRect(
        picture.caption_bbox.x1 * scale,
        picture.caption_bbox.y1 * scale,
        (picture.caption_bbox.x2 - picture.caption_bbox.x1) * scale,
        (picture.caption_bbox.y2 - picture.caption_bbox.y1) * scale
      )
    }

    // Convert to data URL
    pageThumbnail.value = canvas.toDataURL('image/jpeg', 0.85)
    
  } catch (error) {
    console.error('Failed to create page thumbnail:', error)
  } finally {
    pageThumbnailLoading.value = false
  }
}

async function viewImage(picture: Picture) {
  if (!picture.image_path) return
  
  // Extract relative path
  let relativePath = picture.image_path
  if (relativePath.includes('/images/')) {
    relativePath = relativePath.split('/images/')[1]
  }
  
  selectedImagePath.value = relativePath
  
  // Extract page metadata from page_id
  const pageMetadata = parsePageMetadata(picture.page_id)
  selectedPageMetadata.value = {
    date: formatDate(picture.date),
    issue_number: pageMetadata?.issue,
    daily_count: pageMetadata?.daily,
    page_number: pageMetadata?.page,
  }
  
  // Store the picture for the dialog
  selectedPictureForDialog.value = picture
  
  // Fetch all detections for this page to show context (headlines, etc.)
  try {
    const response = await api.get(
      `/layout/${sourceStore.currentSource}/detections`,
      { 
        params: { 
          page_id: picture.page_id,
          run_id: selectedRunId.value 
        } 
      }
    )
    allPageDetections.value = response.data.items
  } catch (error) {
    console.error('Failed to load page detections:', error)
    allPageDetections.value = []
  }
  
  // Load page thumbnail with detection overlay
  loadPageThumbnailWithOverlay(picture)
  
  // Create detection object for OpenSeadragonViewer
  selectedImageDetections.value = [{
    detection_id: picture.detection_id,
    class_name: picture.class_name,
    confidence: picture.confidence,
    bbox: {
      x1: picture.bbox_x1,
      y1: picture.bbox_y1,
      x2: picture.bbox_x2,
      y2: picture.bbox_y2,
    },
  }]
  
  imageViewerDialog.value = true
}



// Navigation logic
const currentPictureIndex = computed(() => {
  if (!selectedPictureForDialog.value) return -1
  return pictures.value.findIndex(p => p.detection_id === selectedPictureForDialog.value?.detection_id)
})

const hasPreviousPicture = computed(() => {
  return currentPictureIndex.value > 0 || currentPage.value > 1
})

const hasNextPicture = computed(() => {
  const isLastOnPage = currentPictureIndex.value === pictures.value.length - 1
  return !isLastOnPage || currentPage.value < totalPages.value
})

function previousPicture() {
  if (currentPictureIndex.value > 0) {
    // Normal case: previous picture is on current page
    const prev = pictures.value[currentPictureIndex.value - 1]
    viewImage(prev)
  } else if (currentPage.value > 1) {
    // Cross-page case: previous picture is on previous page
    currentPage.value--
    
    // Wait for pictures to update
    const unwatch = watch(pictures, (newPictures) => {
      if (newPictures.length > 0) {
        // Open the LAST picture of the previous page
        viewImage(newPictures[newPictures.length - 1])
        unwatch()
      }
    })
  }
}

function nextPicture() {
  const isLastOnPage = currentPictureIndex.value === pictures.value.length - 1
  
  if (!isLastOnPage && currentPictureIndex.value !== -1) {
    // Normal case: next picture is on current page
    const next = pictures.value[currentPictureIndex.value + 1]
    viewImage(next)
  } else if (currentPage.value < totalPages.value) {
    // Cross-page case: next picture is on next page
    currentPage.value++
    
    // Wait for pictures to update
    const unwatch = watch(pictures, (newPictures) => {
      if (newPictures.length > 0) {
        // Open the FIRST picture of the next page
        viewImage(newPictures[0])
        unwatch()
      }
    })
  }
}

// Keyboard navigation
function handleKeydown(e: KeyboardEvent) {
  if (!imageViewerDialog.value) return
  
  if (e.key === 'ArrowLeft') {
    previousPicture()
  } else if (e.key === 'ArrowRight') {
    nextPicture()
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeydown)
})

// Clean up event listener
import { onUnmounted } from 'vue'
onUnmounted(() => {
  window.removeEventListener('keydown', handleKeydown)
})

// Watch for filter changes - loadPictures already called via watch on line 119
// No need for additional logic here as filters trigger reload automatically

// Watch for run changes - clear cache only
watch(selectedRunId, (newRunId, oldRunId) => {
  // Only clear cache if the run actually changed (not initial undefined)
  if (newRunId && newRunId !== oldRunId) {
    // Clear the cropped images cache when switching runs
    clearImageCache()
    // Don't load here - let onMetadataLoaded handle it to avoid double loading
  }
})

function onMetadataLoaded() {
  loadPictures()
}

onMounted(() => {
  if (sourceStore.currentSource) {
    // onMetadataLoaded will be called by ResultsViewer when metadata is loaded
  }
})
</script>

<template>
  <div class="space-y-4 pb-6">
    <!-- Sticky header - compact single row -->
    <div class="sticky top-0 z-10 bg-background py-3 px-6">
      <div class="flex flex-wrap items-start gap-4 mt-2 mb-1">
        <!-- Title -->
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Picture Gallery"
            description="Browse and analyze newspaper picture detections."
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

        <!-- Filters -->
        <FilterBar
          :model-value="filterOptions"
          @update:model-value="updateFilterOptions"
          :filtered-count="totalItems"
          :total-count="backendStats?.total || 0"
        />
      </div>
    </div>

    <!-- Content area -->
    <div class="space-y-6 px-6">
      <!-- Statistics and Charts Section (Collapsible) -->
      <details v-if="pictures.length > 0" class="rounded-lg border bg-card" open>
        <summary class="cursor-pointer p-4 hover:bg-accent/50 transition-colors font-semibold select-none">
          Statistics & Charts
        </summary>
        <div class="p-6 pt-2 space-y-6">
          <!-- Statistics Cards -->
          <StatisticsCards v-if="statistics" :stats="statsCards" :columns="5" />

          <!-- First Row: Timeline and Size Distribution -->
          <div class="grid gap-4 md:grid-cols-2">
            <!-- Timeline Chart -->
            <div v-if="timelineChart.series" class="rounded-lg border bg-card p-4">
              <VChart :option="timelineChart" class="h-[300px]" autoresize />
            </div>
            
            <!-- Size Distribution Chart -->
            <div class="rounded-lg border bg-card p-4">
              <VChart :option="sizeDistributionChart" class="h-[300px]" autoresize />
            </div>
          </div>

          <!-- Second Row: Confidence and Position Distribution -->
          <div class="grid gap-4 md:grid-cols-2">
            <!-- Confidence Distribution -->
            <div class="rounded-lg border bg-card p-4">
              <VChart :option="confidenceDistributionChart" class="h-[300px]" autoresize />
            </div>
            
            <!-- Position Distribution -->
            <div class="rounded-lg border bg-card p-4">
              <VChart :option="positionDistributionChart" class="h-[300px]" autoresize />
            </div>
          </div>
        </div>
      </details>

      <!-- Error Message -->
      <div v-if="errorMessage" class="rounded-lg border border-orange-500 bg-orange-50 dark:bg-orange-950/20 p-4">
        <div class="flex items-start gap-3">
          <svg class="w-5 h-5 text-orange-600 dark:text-orange-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <div class="flex-1">
            <h4 class="font-semibold text-orange-900 dark:text-orange-200 mb-1">Caption Filter Unavailable</h4>
            <p class="text-sm text-orange-800 dark:text-orange-300">{{ errorMessage }}</p>
            <p class="text-xs text-orange-700 dark:text-orange-400 mt-2">
              To enable caption filtering, run: <code class="bg-orange-100 dark:bg-orange-900 px-1 py-0.5 rounded">newspaper-explorer analyze captions enrich --source {{ sourceStore.currentSource }} --run-id YOUR_RUN_ID --classes Caption</code> followed by <code class="bg-orange-100 dark:bg-orange-900 px-1 py-0.5 rounded">newspaper-explorer analyze captions match</code>
            </p>
          </div>
        </div>
      </div>

      <!-- Loading state (only for initial load) -->
      <div v-if="loading && pictures.length === 0" class="text-center py-12">
        <p class="text-muted-foreground">Loading pictures...</p>
      </div>

      <!-- Picture Gallery -->
      <div v-else-if="pictures.length > 0 || loading" class="rounded-lg border bg-card p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold">
            Picture Gallery ({{ pictures.length.toLocaleString() }} pictures)
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
              Page {{ currentPage }} of {{ totalPages }}
            </div>
          </div>
        </div>

        <!-- Picture Grid (2 rows max) -->
        <div class="grid gap-6 grid-cols-4">
          <template v-if="loading">
            <!-- Loading placeholders to prevent layout shift -->
            <div
              v-for="i in pageSize"
              :key="`placeholder-${i}`"
              class="rounded-lg border bg-card overflow-hidden"
            >
              <div class="aspect-[4/3] bg-muted animate-pulse"></div>
              <div class="p-4 space-y-2">
                <div class="h-4 bg-muted rounded animate-pulse"></div>
                <div class="h-3 bg-muted rounded animate-pulse"></div>
                <div class="h-3 bg-muted rounded w-3/4 animate-pulse"></div>
              </div>
            </div>
          </template>
          <template v-else>
            <PictureCard
              v-for="picture in pictures"
              :key="picture.detection_id"
              :picture="picture"
              :cropped-image="croppedImages[picture.detection_id]"
              :cropped-caption="croppedCaptions[`caption_${picture.detection_id}`]"
              :is-loading-image="cropLoadingImages.has(picture.detection_id)"
              :is-loading-caption="cropLoadingCaptions.has(`caption_${picture.detection_id}`)"
              @click="viewImage"
            />
          </template>
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
      <div v-else-if="!loading" class="rounded-lg border bg-card p-12 text-center space-y-4">
        <svg
          class="mx-auto h-12 w-12 text-muted-foreground"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <div>
          <h3 class="text-lg font-semibold">No pictures found</h3>
          <p class="text-muted-foreground mt-2">
            Run layout detection first to extract picture regions from newspaper pages.
          </p>
        </div>
      </div>
    </div>

    <!-- Picture Detail Dialog -->
    <PictureDetailDialog
      :open="imageViewerDialog"
      :picture="selectedPictureForDialog"
      :source="sourceStore.currentSource || ''"
      :image-path="selectedImagePath"
      :page-metadata="selectedPageMetadata"
      :all-page-detections="allPageDetections"
      :cropped-images="croppedImages"
      :cropped-captions="croppedCaptions"
      :crop-loading-images="cropLoadingImages"
      :crop-loading-captions="cropLoadingCaptions"
      :page-thumbnail="pageThumbnail"
      :page-thumbnail-loading="pageThumbnailLoading"
      :has-previous="hasPreviousPicture"
      :has-next="hasNextPicture"
      @close="imageViewerDialog = false"
      @previous="previousPicture"
      @next="nextPicture"
    />

    <!-- Thumbnail Gallery Dialog -->
    <ThumbnailGalleryDialog
      :open="thumbnailGalleryOpen"
      :pictures="galleryPictures"
      :cropped-images="croppedImages"
      :loading="galleryLoading"
      :current-page="galleryPage"
      :total-pages="galleryTotalPages"
      :total-items="galleryTotalItems"
      @close="thumbnailGalleryOpen = false"
      @select="selectFromGallery"
      @page-change="goToGalleryPage"
      @load-image="loadPictureCrop"
    />
  </div>
</template>
