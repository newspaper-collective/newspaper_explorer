<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSourceStore } from '@/stores/source'
import api from '@/lib/api'
import type { SearchResponse } from '@/types'
import { type SearchFilterOptions } from '@/components/SearchFilterBar.vue'
import { type SearchResult } from '@/components/SearchHitCard.vue'
import OpenSeadragonViewer from '@/components/OpenSeadragonViewer.vue'
import ResultsViewer from '@/components/ResultsViewer.vue'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import PaginationControls from '@/components/PaginationControls.vue'
import { usePagination } from '@/lib/composables/usePagination'
import { getFullImageUrl, parsePageMetadata, formatDate } from '@/lib/composables/useImageUtils'
import { X, ExternalLink, Grid3x3, List } from 'lucide-vue-next'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

const sourceStore = useSourceStore()
const router = useRouter()
const results = ref<SearchResponse | null>(null)
const loading = ref(false)
const selectedRunId = ref<string | null>(null)

// View mode
const viewMode = ref<'grid' | 'list'>('list')

// Filters
const filterOptions = ref<SearchFilterOptions>({
  query: '',
})

// Sorting
const sortOrder = ref<'asc' | 'desc'>('desc') // Default to newest first

// Pagination
const {
  currentPage,
  totalItems,
  totalPages,
  pageSize,
  goToPage
} = usePagination({ itemsPerPage: 50 })

// Watch for page changes
watch(currentPage, () => {
  search()
})

// Watch for sort order changes
watch(sortOrder, () => {
  currentPage.value = 1
  search()
})

// Watch for run changes
watch(selectedRunId, () => {
  // Reset pagination and search when run changes
  currentPage.value = 1
  if (filterOptions.value.query) {
    search()
  }
})

// Initialize date filters from source stats
watch(() => sourceStore.sourceStats, (stats) => {
  if (stats && stats.years_available && stats.years_available.length > 0) {
    // We could set defaults here if desired, but usually we leave them empty for "all time"
    // unless the user wants to enforce the source range.
    // For now, we'll just let the user pick.
  }
}, { immediate: true })

async function search() {
  if (!sourceStore.currentSource || !filterOptions.value.query.trim()) return

  loading.value = true
  try {
    const params: any = {
      query: filterOptions.value.query,
      run_id: selectedRunId.value,
      sort_order: sortOrder.value,
      pagination: {
        page: currentPage.value,
        page_size: pageSize.value
      },
    }

    if (sourceStore.startDate || sourceStore.endDate) {
      params.date_filter = {
        start_date: sourceStore.startDate || null,
        end_date: sourceStore.endDate || null,
      }
    }

    const response = await api.post(`/search/${sourceStore.currentSource}/`, params)
    results.value = response.data
    totalItems.value = response.data.total
  } catch (error) {
    console.error('Search failed:', error)
    results.value = null
    totalItems.value = 0
  } finally {
    loading.value = false
  }
}

// Group results by page
const groupedResults = computed(() => {
  if (!results.value?.results) return []

  const groups = new Map<string, { page_id: string; date: string; results: SearchResult[]; metadata: any }>()

  results.value.results.forEach((result: SearchResult) => {
    if (!groups.has(result.page_id)) {
      const metadata = parsePageMetadata(result.page_id)
      groups.set(result.page_id, {
        page_id: result.page_id,
        date: result.date,
        results: [],
        metadata,
      })
    }
    groups.get(result.page_id)!.results.push(result)
  })

  return Array.from(groups.values())
})

// View Page Logic
const viewerOpen = ref(false)
const selectedResult = ref<SearchResult | null>(null)
const viewerImageUrl = ref('')
const scaledTextLines = ref<any[]>([])

const currentIssueId = computed(() => {
  if (!selectedResult.value) return null
  const metadata = parsePageMetadata(selectedResult.value.page_id)
  if (!metadata) return null
  // Reconstruct issue_id: {source}_{date}_{issue}_{daily}
  return `${metadata.source}_${metadata.date}_${metadata.issue}_${metadata.daily}`
})

const currentPageNumber = computed(() => {
  if (!selectedResult.value) return null
  const metadata = parsePageMetadata(selectedResult.value.page_id)
  return metadata?.page ? parseInt(metadata.page) : null
})

function viewPage(result: SearchResult) {
  if (!sourceStore.currentSource) return

  selectedResult.value = result

  if (result.image_path) {
    const url = getFullImageUrl(result.image_path, sourceStore.currentSource)
    if (url) {
      viewerImageUrl.value = url
      // Fetch page details and scale coordinates like IssueReader does
      fetchPageDetailsForViewer(result.page_id, result)
      viewerOpen.value = true
    } else {
      console.error('Could not generate image URL from path', result.image_path)
    }
  } else {
    // Fallback to fetching page details if image_path is missing (backward compatibility)
    fetchPageDetails(result.page_id)
  }
}

async function fetchPageDetailsForViewer(pageId: string, result: SearchResult) {
  if (!sourceStore.currentSource) return

  try {
    // Get page info including ALTO and image dimensions
    const response = await api.get(`/data/${sourceStore.currentSource}/pages`, {
      params: {
        page_id: pageId,
        page_size: 1
      }
    })

    if (response.data && response.data.length > 0) {
      const page = response.data[0]

      // Scale coordinates from ALTO space to image space (like IssueReader)
      const altoWidth = page.alto_width
      const altoHeight = page.alto_height
      const imageWidth = page.image_width
      const imageHeight = page.image_height

      const scaleX = (altoWidth && imageWidth) ? imageWidth / altoWidth : 1
      const scaleY = (altoHeight && imageHeight) ? imageHeight / altoHeight : 1

      // Apply scaling to the text line coordinates
      if (result.x !== undefined && result.y !== undefined) {
        scaledTextLines.value = [{
          text: result.text,
          x: Math.round(result.x * scaleX),
          y: Math.round(result.y * scaleY),
          width: Math.round((result.width || 0) * scaleX),
          height: Math.round((result.height || 0) * scaleY),
          line_id: result.text_block_id
        }]
      } else {
        scaledTextLines.value = []
      }
    }
  } catch (error) {
    console.error('Failed to fetch page details:', error)
  }
}

async function fetchPageDetails(pageId: string) {
  if (!sourceStore.currentSource) return

  try {
    const response = await api.get(`/layout/${sourceStore.currentSource}/detections`, {
      params: {
        page_id: pageId,
        page_size: 1
      }
    })

    if (response.data.items && response.data.items.length > 0) {
      const imagePath = response.data.items[0].image_path
      const url = getFullImageUrl(imagePath, sourceStore.currentSource)
      if (url) {
        viewerImageUrl.value = url
        viewerOpen.value = true
      }
    }
  } catch (error) {
    console.error('Failed to fetch page details:', error)
  }
}

function closeViewer() {
  viewerOpen.value = false
  selectedResult.value = null
  viewerImageUrl.value = ''
  scaledTextLines.value = []
}

function openInIssueReader() {
  if (!currentIssueId.value) return

  router.push({
    name: 'issue',
    params: { issueId: currentIssueId.value },
    query: { page: (currentPageNumber.value || 1).toString() }
  })
}

function onMetadataLoaded() {
  // Optional: trigger search if query exists
  if (filterOptions.value.query) {
    search()
  }
}
</script>

<template>
  <div class="space-y-6 pb-6 h-full flex flex-col">
    <!-- Header -->
    <div class="px-4 pt-4">
      <div class="flex flex-wrap items-start gap-4 mb-6">
        <!-- Title -->
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Search"
            description="Search through newspaper text content"
            icon="search"
          />
        </div>

        <!-- Results Selector -->
        <div class="flex-1 min-w-[200px] self-stretch">
          <ResultsViewer
            v-if="sourceStore.currentSource"
            ref="resultsViewer"
            :source="sourceStore.currentSource"
            analysis-type="text"
            v-model:run-id="selectedRunId"
            @loaded="onMetadataLoaded"
          />
        </div>

        <!-- Search Box -->
        <div class="rounded-lg border bg-card p-3 flex-1 min-w-[300px] self-stretch">
          <div class="flex flex-col gap-1.5 h-full">
            <!-- Header -->
            <label class="text-sm font-medium mb-0.5">Search Input</label>

            <!-- Search Input and Button -->
            <div class="flex">
              <Input
                v-model="filterOptions.query"
                @keydown.enter="currentPage = 1; search()"
                type="text"
                placeholder="Search text content..."
                class="flex-1 rounded-r-none focus:z-10"
              />
              <Button
                @click="currentPage = 1; search()"
                :disabled="loading || !filterOptions.query"
                class="rounded-l-none"
              >
                Search
              </Button>
            </div>

            <!-- Results Count and Sort Options -->
            <div class="flex items-center justify-between text-sm text-muted-foreground">
              <span v-if="totalItems > 0">
                {{ totalItems.toLocaleString() }} results • {{ groupedResults.length }} pages
              </span>
              <span v-else-if="!loading && results">No results</span>
              <span v-else>&nbsp;</span>

              <div class="flex items-center gap-4">
                <!-- Sort Selector -->
                <div class="flex items-center gap-2">
                  <label class="text-sm text-muted-foreground">Sort:</label>
                  <Select v-model="sortOrder">
                    <SelectTrigger class="w-[140px] h-8 text-sm">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="desc">Newest First</SelectItem>
                      <SelectItem value="asc">Oldest First</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <!-- View Mode Toggle -->
                <div class="flex items-center gap-0.5 border rounded-md p-0.5">
                  <Button
                    @click="viewMode = 'list'"
                    :variant="viewMode === 'list' ? 'default' : 'ghost'"
                    size="icon"
                    class="h-6 w-6"
                    title="List view"
                  >
                    <List class="h-3.5 w-3.5" />
                  </Button>
                  <Button
                    @click="viewMode = 'grid'"
                    :variant="viewMode === 'grid' ? 'default' : 'ghost'"
                    size="icon"
                    class="h-6 w-6"
                    title="Grid view"
                  >
                    <Grid3x3 class="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 flex-1 overflow-auto">
      <div v-if="loading && !results" class="text-center py-12">
        <p class="text-muted-foreground">Searching...</p>
      </div>

      <div v-else-if="results && results.results.length > 0">
        <!-- List View -->
        <div v-if="viewMode === 'list'" class="space-y-3">
          <div
            v-for="group in groupedResults"
            :key="group.page_id"
            class="border rounded-lg bg-card overflow-hidden"
            :class="{ 'compact-group': group.results.length <= 2 }"
          >
            <!-- Page Header (compact for small groups) -->
            <div
              class="bg-muted/30 border-b"
              :class="group.results.length <= 2 ? 'px-3 py-2' : 'px-4 py-3'"
            >
              <div class="flex items-center justify-between gap-4">
                <div class="flex items-center gap-3 min-w-0 flex-1">
                  <div class="font-medium text-sm whitespace-nowrap">{{ formatDate(group.date) }}</div>
                  <div class="text-xs text-muted-foreground truncate">
                    {{ group.metadata ? `Issue ${group.metadata.issue} • Daily ${group.metadata.daily} • Page ${group.metadata.page}` : group.page_id }}
                  </div>
                </div>
                <div class="text-xs text-muted-foreground whitespace-nowrap">
                  {{ group.results.length }} {{ group.results.length === 1 ? 'result' : 'results' }}
                </div>
              </div>
            </div>

            <!-- Results for this page (adaptive columns based on count) -->
            <div :class="group.results.length <= 2 ? 'p-3' : 'p-4'">
              <div
                class="grid gap-3"
                :class="{
                  'grid-cols-1': group.results.length <= 2,
                  'md:grid-cols-2 lg:grid-cols-3': group.results.length > 2
                }"
              >
                <div
                  v-for="result in group.results"
                  :key="result.text_block_id"
                  class="border rounded-lg hover:shadow-md transition-shadow bg-background"
                  :class="group.results.length <= 2 ? 'p-2.5' : 'p-3'"
                >
                  <div class="flex justify-between items-start mb-2">
                    <div class="text-xs text-muted-foreground font-mono truncate flex-1 mr-2">{{ result.text_block_id }}</div>
                    <Button
                      @click="viewPage(result)"
                      variant="outline"
                      size="sm"
                      class="h-6 text-xs flex-shrink-0"
                    >
                      View Page
                    </Button>
                  </div>

                  <div
                    class="text-sm leading-relaxed font-serif bg-muted/30 rounded border border-border/50"
                    :class="group.results.length <= 2 ? 'p-2.5' : 'p-3'"
                  >
                    <div v-if="result.highlights && result.highlights.length > 0">
                      <span v-for="(highlight, idx) in result.highlights" :key="idx">
                        <span v-if="idx > 0">...<br/></span>
                        <span>...{{ highlight }}...</span>
                      </span>
                    </div>
                    <div v-else>
                      {{ result.text.length > 300 ? result.text.substring(0, 300) + '...' : result.text }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Grid View -->
        <div v-else class="grid gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <div
            v-for="group in groupedResults"
            :key="group.page_id"
            class="border rounded-lg bg-card overflow-hidden flex flex-col"
          >
            <!-- Compact Page Header -->
            <div class="bg-muted/30 border-b px-3 py-2 flex-shrink-0">
              <div class="flex items-center justify-between gap-2">
                <div class="font-medium text-sm">{{ formatDate(group.date) }}</div>
                <div class="text-xs text-muted-foreground whitespace-nowrap">
                  {{ group.results.length }} {{ group.results.length === 1 ? 'result' : 'results' }}
                </div>
              </div>
              <div class="text-xs text-muted-foreground truncate">
                {{ group.metadata ? `Iss. ${group.metadata.issue} • D. ${group.metadata.daily} • P. ${group.metadata.page}` : group.page_id }}
              </div>
            </div>

            <!-- Results (single column in grid view) -->
            <div class="p-3 space-y-2 flex-1 overflow-auto">
              <div
                v-for="result in group.results"
                :key="result.text_block_id"
                class="border rounded-lg p-2 hover:shadow-md transition-shadow bg-background"
              >
                <div class="flex justify-between items-start mb-1.5 gap-2">
                  <div class="text-xs text-muted-foreground font-mono truncate flex-1">{{ result.text_block_id }}</div>
                  <Button
                    @click="viewPage(result)"
                    variant="outline"
                    size="sm"
                    class="h-5 px-2 text-xs flex-shrink-0"
                  >
                    View
                  </Button>
                </div>

                <div class="text-xs leading-relaxed font-serif bg-muted/30 p-2 rounded border border-border/50 line-clamp-4">
                  <div v-if="result.highlights && result.highlights.length > 0">
                    <span v-for="(highlight, idx) in result.highlights" :key="idx">
                      <span v-if="idx > 0">... </span>
                      <span>...{{ highlight }}...</span>
                    </span>
                  </div>
                  <div v-else>
                    {{ result.text.length > 200 ? result.text.substring(0, 200) + '...' : result.text }}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Pagination -->
        <PaginationControls
          :current-page="currentPage"
          :total-pages="totalPages"
          :loading="loading"
          @update:current-page="goToPage"
        />
      </div>

      <div v-else-if="results" class="text-center py-12">
        <p class="text-muted-foreground">No results found.</p>
      </div>

      <div v-else class="text-center py-12">
        <p class="text-muted-foreground">Enter a query to start searching.</p>
      </div>
    </div>

    <!-- Viewer Dialog (LayoutView style) -->
    <Teleport to="body">
      <div
        v-if="viewerOpen"
        class="fixed inset-0 z-[100] flex items-center justify-center bg-overlay-heavy"
        @click="closeViewer"
      >
        <div
          class="bg-card rounded-lg shadow-lg w-[70vw] max-w-[1000px] h-[95vh] overflow-hidden m-4 flex flex-col"
          @click.stop
        >
        <div class="flex items-center justify-between p-4 border-b bg-muted/50">
          <div v-if="selectedResult" class="space-y-1">
            <p class="font-semibold text-base">
              {{ formatDate(selectedResult.date) }}
            </p>
            <p class="text-xs text-muted-foreground">
              {{ selectedResult.page_id }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <Button
              v-if="currentIssueId"
              @click="openInIssueReader"
              variant="outline"
              size="sm"
            >
              <ExternalLink class="h-3 w-3 mr-2" />
              Open in Issue Reader
            </Button>
            <Button
              @click="closeViewer"
              variant="ghost"
              size="icon"
            >
              <X class="w-5 h-5" />
            </Button>
          </div>
        </div>
        <div class="flex-1 relative">
          <OpenSeadragonViewer
            v-if="viewerImageUrl"
            :image-url="viewerImageUrl"
            :current-page="1"
            :total-pages="1"
            :text-lines="scaledTextLines"
            :highlighted-line-id="selectedResult?.text_block_id"
          />
        </div>
      </div>
      </div>
    </Teleport>
  </div>
</template>
