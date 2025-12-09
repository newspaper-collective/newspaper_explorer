<script setup lang="ts">
import { ref, onMounted, watch, computed } from 'vue'
import { useSourceStore } from '@/stores/source'
import { useRoute, useRouter } from 'vue-router'
import api from '@/lib/api'
import { Home, Eye, EyeOff, BarChart3 } from 'lucide-vue-next'
import OpenSeadragonViewer from './OpenSeadragonViewer.vue'
import AnalysisModal from './AnalysisModal.vue'

interface Props {
  issueId: string
  initialPage?: number
  showBackButton?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  initialPage: 1,
  showBackButton: true,
})

const emit = defineEmits<{
  back: []
  backToMonth: [year: number, month: number]
}>()

const sourceStore = useSourceStore()
const route = useRoute()
const router = useRouter()
const currentPage = ref(props.initialPage)
const loading = ref(false)
const showOverlays = ref(false)
const showLayoutOverlays = ref(false)
const highlightedLineId = ref<string | null>(null)
const showAnalysisSidebar = ref(false)

// Issue metadata
const metadata = ref<any>(null)
const pages = ref<any[]>([])
const pageContent = ref<any>(null)
const textLines = ref<any[]>([])
const layoutRegions = ref<any[]>([])
const availableLayoutSets = ref<string[]>([])
const selectedLayoutSet = ref<string | null>(null)

const monthNames = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

const sourceTitle = computed(() => {
  return sourceStore.sourceInfo?.metadata?.newspaper_title || 'Collection'
})

const monthName = computed(() => {
  if (metadata.value?.month) {
    return monthNames[metadata.value.month - 1]
  }
  return null
})

const displayedTextLines = computed(() => {
  return showOverlays.value ? textLines.value : []
})

const displayedLayoutRegions = computed(() => {
  return showLayoutOverlays.value ? layoutRegions.value : []
})

async function loadIssueMetadata() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    // Get all pages for this issue
    const response = await api.get(
      `/data/${sourceStore.currentSource}/pages`,
      { params: { issue_id: props.issueId, page_size: 1000 } }
    )
    // Sort pages by page_number to ensure correct order
    pages.value = response.data.sort((a: any, b: any) => a.page_number - b.page_number)

    if (pages.value.length > 0) {
      const firstPage = pages.value[0]
      const issueId = firstPage.issue_id || props.issueId

      // Extract issue_number and daily_count from issue_id
      // Format: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
      const issueParts = issueId.split('_')
      const issueNumber = issueParts.length >= 3 ? issueParts[issueParts.length - 2] : null
      const dailyCount = issueParts.length >= 4 ? issueParts[issueParts.length - 1] : null

      // Extract year and month from date
      const date = new Date(firstPage.date)
      const year = date.getFullYear()
      const month = date.getMonth() + 1

      metadata.value = {
        title: firstPage.newspaper_title,
        date: firstPage.date,
        pageCount: pages.value.length,
        issueNumber,
        dailyCount,
        year,
        month,
      }

      // Load first page content
      await loadPageContent(currentPage.value)
    }
  } catch (error) {
    console.error('Failed to load issue:', error)
  } finally {
    loading.value = false
  }
}

async function loadPageContent(pageNum: number) {
  if (!sourceStore.currentSource || !pages.value[pageNum - 1]) return

  loading.value = true
  try {
    const page = pages.value[pageNum - 1]

    // Get text lines for this page (individual lines with bounding boxes)
    const linesResponse = await api.get(
      `/data/${sourceStore.currentSource}/lines`,
      { params: { page_id: page.page_id, page_size: 5000 } }
    )

    // Scale coordinates from ALTO to image dimensions
    const altoWidth = page.alto_width
    const altoHeight = page.alto_height
    const imageWidth = page.image_width
    const imageHeight = page.image_height

    const scaleX = (altoWidth && imageWidth) ? imageWidth / altoWidth : 1
    const scaleY = (altoHeight && imageHeight) ? imageHeight / altoHeight : 1

    // Apply scaling to all lines
    const scaledLines = linesResponse.data.map((line: any) => ({
      ...line,
      x: Math.round(line.x * scaleX),
      y: Math.round(line.y * scaleY),
      width: Math.round(line.width * scaleX),
      height: Math.round(line.height * scaleY),
    }))

    textLines.value = scaledLines

    // Group lines by text_block_id for display
    const blockMap = new Map()
    scaledLines.forEach((line: any) => {
      if (!blockMap.has(line.text_block_id)) {
        blockMap.set(line.text_block_id, {
          text_block_id: line.text_block_id,
          lines: [],
          text: '',
        })
      }
      blockMap.get(line.text_block_id).lines.push(line)
    })

    // Create blocks with concatenated text
    const blocks = Array.from(blockMap.values()).map(block => ({
      ...block,
      text: block.lines.map((l: any) => l.text).join(' '),
    }))

    pageContent.value = {
      page,
      blocks,
      imageUrl: page.image_url,
    }

    // Load available layout analysis results
    await loadLayoutResults()
  } catch (error) {
    console.error('Failed to load page content:', error)
  } finally {
    loading.value = false
  }
}

async function loadLayoutResults() {
  if (!sourceStore.currentSource || !pageContent.value?.page?.page_id) return

  try {
    const response = await api.get(
      `/data/${sourceStore.currentSource}/page-analysis/${pageContent.value.page.page_id}`
    )

    const layoutData = response.data.layout || {}
    availableLayoutSets.value = Object.keys(layoutData)

    // Auto-select first available set or keep current selection
    if (availableLayoutSets.value.length > 0) {
      if (!selectedLayoutSet.value) {
        selectedLayoutSet.value = availableLayoutSets.value[0]
      }
      // Always update regions when page changes, even if selection didn't change
      updateLayoutRegions()
    } else {
      layoutRegions.value = []
    }
  } catch (error) {
    console.error('Failed to load layout results:', error)
  }
}

async function updateLayoutRegions() {
  if (!sourceStore.currentSource || !pageContent.value?.page?.page_id || !selectedLayoutSet.value) {
    layoutRegions.value = []
    return
  }

  try {
    const response = await api.get(
      `/data/${sourceStore.currentSource}/page-analysis/${pageContent.value.page.page_id}`
    )

    const layoutData = response.data.layout || {}
    const regions = layoutData[selectedLayoutSet.value] || []

    // Layout detection coordinates are already in image pixel space, no scaling needed
    const filteredRegions = regions.filter((region: any) => {
      // Check if bbox exists as an object with x1, or if coordinates are at top level
      const hasValidBbox = (region.bbox && typeof region.bbox === 'object' && 'x1' in region.bbox) ||
                          ('x1' in region && 'y1' in region && 'x2' in region && 'y2' in region)
      return hasValidBbox
    })

    layoutRegions.value = filteredRegions.map((region: any) => {
      // Handle both nested bbox and top-level coordinates
      let bbox
      if (region.bbox && typeof region.bbox === 'object') {
        bbox = region.bbox
      } else if ('x1' in region && 'y1' in region) {
        bbox = region
      } else {
        console.error('Region has no valid bbox structure:', region)
        return null
      }

      // No scaling - layout detections are already in image pixel coordinates
      return {
        ...region,
        bbox: {
          x1: bbox.x1,
          y1: bbox.y1,
          x2: bbox.x2,
          y2: bbox.y2,
        }
      }
    }).filter((r: any) => r !== null)
  } catch (error) {
    console.error('Failed to update layout regions:', error)
    layoutRegions.value = []
  }
}

// Watch for layout set changes
watch(selectedLayoutSet, () => {
  updateLayoutRegions()
})

function handleLineHover(lineId: string | null) {
  highlightedLineId.value = lineId
}

function handleLineClick(lineId: string) {
  highlightedLineId.value = lineId
  // Scroll to the line in the text panel
  const lineElement = document.querySelector(`[data-line-id="${lineId}"]`) as HTMLElement
  if (lineElement) {
    // Find the scrollable container (the text content div)
    const scrollContainer = lineElement.closest('.overflow-y-auto') as HTMLElement
    if (scrollContainer) {
      const containerTop = scrollContainer.offsetTop
      const containerHeight = scrollContainer.clientHeight
      const lineTop = lineElement.offsetTop
      const lineHeight = lineElement.clientHeight

      // Calculate scroll position to center the line in the container
      const scrollTo = lineTop - containerTop - (containerHeight / 2) + (lineHeight / 2)

      scrollContainer.scrollTo({ top: scrollTo, behavior: 'smooth' })
    } else {
      // Fallback to scrollIntoView if container not found
      lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }
}

function changePage(delta: number) {
  const newPage = currentPage.value + delta
  if (newPage >= 1 && newPage <= pages.value.length) {
    currentPage.value = newPage
    loadPageContent(newPage)
    updateUrlWithPage(newPage)
  }
}

function updateUrlWithPage(page: number) {
  // Update URL query parameter without triggering navigation
  const query = { ...route.query, page: page.toString() }
  router.replace({ query })
}

function goToBrowse() {
  router.push({ name: 'browse' })
}

function goToYear() {
  if (metadata.value?.year) {
    router.push({
      name: 'browse',
      query: { year: metadata.value.year.toString() }
    })
  } else {
    router.push({ name: 'browse' })
  }
}

function goBackToMonth() {
  if (metadata.value?.year && metadata.value?.month) {
    router.push({
      name: 'browse',
      query: { year: metadata.value.year.toString(), month: metadata.value.month.toString() }
    })
  } else {
    emit('back')
  }
}

function goBackToGallery() {
  router.push({ name: 'issue-gallery', params: { issueId: props.issueId } })
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('de-DE')
}

watch(() => props.issueId, () => {
  currentPage.value = props.initialPage
  loadIssueMetadata()
})

// Watch for page changes in route query
watch(() => route.query.page, (newPage) => {
  if (newPage && typeof newPage === 'string') {
    const pageNum = parseInt(newPage, 10)
    if (!isNaN(pageNum) && pageNum !== currentPage.value) {
      currentPage.value = pageNum
      loadPageContent(pageNum)
    }
  }
})

onMounted(() => {
  // Check if page parameter is in URL
  if (route.query.page && typeof route.query.page === 'string') {
    const pageNum = parseInt(route.query.page, 10)
    if (!isNaN(pageNum)) {
      currentPage.value = pageNum
    }
  }
  loadIssueMetadata()
})
</script>

<template>
  <div class="space-y-6 px-4 pb-4">
    <!-- Breadcrumb Navigation (matching BrowseView) -->
    <div class="mt-4">
      <div class="flex items-center gap-2 flex-wrap">
        <button
          v-if="showBackButton"
        @click="goToBrowse"
        class="flex items-center gap-2 text-lg font-medium text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
      >
        <Home class="h-5 w-5" />
        {{ sourceTitle }}
      </button>
      <template v-if="metadata">
        <span class="text-muted-foreground text-lg">›</span>
        <button
          @click="goToYear"
          class="text-lg font-medium text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
        >
          {{ metadata.year }}
        </button>
        <span class="text-muted-foreground text-lg">›</span>
        <button
          @click="goBackToMonth"
          class="text-lg font-medium text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
        >
          {{ monthName }}
        </button>
        <span class="text-muted-foreground text-lg">›</span>
        <button
          @click="goBackToGallery"
          class="text-lg font-medium text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
        >
          {{ formatDate(metadata.date) }}
        </button>
        <span class="text-muted-foreground text-lg">›</span>
        <span class="text-lg font-medium text-foreground">
          Page {{ currentPage }}
        </span>
      </template>
      </div>
    </div>

    <!-- Issue Metadata Subtitle -->
    <div v-if="metadata" class="flex items-center justify-between gap-4">
      <div class="text-sm text-muted-foreground">
        {{ formatDate(metadata.date) }}
        <template v-if="metadata.issueNumber">
          • Issue {{ metadata.issueNumber }}
        </template>
        <template v-if="metadata.dailyCount">
          ({{ metadata.dailyCount }} of day)
        </template>
        • {{ metadata.pageCount }} {{ metadata.pageCount === 1 ? 'page' : 'pages' }}
      </div>

      <div class="flex items-center gap-4">
        <!-- Layout Overlay Toggle with Dropdown (hidden when no detections) -->
        <div v-if="availableLayoutSets.length > 0" class="flex items-center gap-2">
          <select
            v-if="showLayoutOverlays"
            v-model="selectedLayoutSet"
            class="px-2 py-2 text-xs"
            title="Select layout result set"
          >
            <option v-for="set in availableLayoutSets" :key="set" :value="set">
              {{ set.split('_').slice(0, 2).join(' ') }}
            </option>
          </select>

          <button
            @click="showLayoutOverlays = !showLayoutOverlays"
            class="flex items-center gap-2 px-3 py-2 text-sm rounded-lg hover:bg-accent transition-colors"
            :class="{ 'bg-accent': showLayoutOverlays }"
            title="Toggle layout overlays on image"
          >
            <component :is="showLayoutOverlays ? Eye : EyeOff" class="h-4 w-4" />
            <span>Layout</span>
          </button>
        </div>

        <!-- Text Overlay Toggle -->
        <button
          @click="showOverlays = !showOverlays"
          class="flex items-center gap-2 px-3 py-2 text-sm rounded-lg hover:bg-accent transition-colors"
          :class="{ 'bg-accent': showOverlays }"
          title="Toggle text overlays on image"
        >
          <component :is="showOverlays ? Eye : EyeOff" class="h-4 w-4" />
          <span>Text</span>
        </button>

        <!-- Analysis Sidebar Toggle -->
        <button
          @click="showAnalysisSidebar = !showAnalysisSidebar"
          class="flex items-center gap-2 px-3 py-2 text-sm rounded-lg hover:bg-accent transition-colors"
          :class="{ 'bg-accent': showAnalysisSidebar }"
          title="Toggle analysis sidebar"
        >
          <BarChart3 class="h-4 w-4" />
          <span>Analysis</span>
        </button>

        <!-- Page Navigation -->
        <div class="flex items-center gap-2">
          <button
            @click="changePage(-1)"
            :disabled="currentPage === 1"
            class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Previous page"
          >
            ←
          </button>
          <span class="text-sm font-medium min-w-[60px] text-center">
            {{ currentPage }} / {{ pages.length }}
          </span>
          <button
            @click="changePage(1)"
            :disabled="currentPage === pages.length"
            class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Next page"
          >
            →
          </button>
        </div>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading && !metadata" class="text-center py-12">
      <p class="text-muted-foreground">Loading issue...</p>
    </div>

    <!-- Page Content -->
    <div v-else-if="pageContent" class="grid gap-3" :class="showAnalysisSidebar ? 'grid-cols-1 lg:grid-cols-3' : 'grid-cols-1 lg:grid-cols-2'">
      <!-- Image with OpenSeadragon Viewer -->
      <div v-if="pageContent.imageUrl" class="rounded-lg border bg-black overflow-hidden h-[calc(100vh-13rem)]">
        <OpenSeadragonViewer
          :image-url="pageContent.imageUrl"
          :current-page="currentPage"
          :total-pages="pages.length"
          :text-lines="displayedTextLines"
          :detections="displayedLayoutRegions"
          :highlighted-line-id="highlightedLineId"
          @change-page="changePage"
          @line-click="handleLineClick"
          @line-hover="handleLineHover"
        />
      </div>

      <!-- Text Content - Matched Height -->
      <div class="rounded-lg border bg-card p-4 space-y-4 overflow-y-auto h-[calc(100vh-13rem)]">
        <div
          v-for="block in pageContent.blocks"
          :key="block.text_block_id"
          class="space-y-1"
        >
          <div
            v-for="line in block.lines"
            :key="line.line_id"
            :data-line-id="line.line_id"
            class="px-2 py-1 rounded transition-colors cursor-pointer"
            :class="{
              'bg-highlight text-highlight-foreground font-medium': highlightedLineId === line.line_id,
              'hover:bg-accent': highlightedLineId !== line.line_id,
            }"
            @mouseenter="handleLineHover(line.line_id)"
            @mouseleave="handleLineHover(null)"
            @click="handleLineClick(line.line_id)"
          >
            <p class="text-sm leading-relaxed">{{ line.text }}</p>
          </div>
        </div>
        <div v-if="!pageContent.blocks || pageContent.blocks.length === 0" class="text-center py-8">
          <p class="text-muted-foreground">No text content available</p>
        </div>
      </div>

      <!-- Analysis Sidebar -->
      <div v-show="showAnalysisSidebar" class="rounded-lg border bg-card overflow-hidden h-[calc(100vh-13rem)]">
        <AnalysisModal
          :is-open="showAnalysisSidebar"
          :page-id="pageContent?.page?.page_id || null"
          :source-name="sourceStore.currentSource || ''"
          :sidebar-mode="true"
          @close="showAnalysisSidebar = false"
        />
      </div>
    </div>

    <!-- No Data -->
    <div v-else class="rounded-lg border bg-card p-8 text-center">
      <p class="text-muted-foreground">Issue not found</p>
    </div>
  </div>
</template>
