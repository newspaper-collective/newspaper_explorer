<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-[100] flex items-center justify-center bg-overlay-medium" @click="$emit('close')">
      <div class="relative w-[90vw] h-[90vh] bg-card rounded-lg shadow-xl flex flex-col" @click.stop>
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b">
        <div>
          <h2 class="text-xl font-semibold">Page Gallery</h2>
          <p class="text-sm text-muted-foreground">
            {{ totalItems.toLocaleString() }} pages total • Page {{ currentPage }} of {{ totalPages }}
          </p>
        </div>
        <button
          @click="$emit('close')"
          class="p-2 rounded-lg hover:bg-accent transition-colors"
          title="Close"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <!-- Scrollable Grid -->
      <div ref="gridRef" class="flex-1 overflow-y-auto p-6">
        <div class="grid gap-4 grid-cols-6 xl:grid-cols-8">
          <div
            v-for="page in pages"
            :key="page.page_id"
            @click="$emit('select', page)"
            class="group cursor-pointer rounded-lg border bg-card hover:border-primary transition-all overflow-hidden"
          >
            <!-- Page thumbnail with overlays -->
            <div class="aspect-[3/4] relative">
              <!-- Canvas for image with overlays -->
              <canvas
                :id="`canvas-${page.page_id}`"
                class="w-full h-full"
              />

              <!-- Hover Overlay -->
              <div class="absolute inset-0 bg-overlay-muted opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span class="text-white text-sm font-medium">View Details</span>
              </div>
            </div>

            <!-- Info -->
            <div class="p-2 border-t">
              <p class="text-xs text-muted-foreground truncate" :title="page.metadata.date || 'Unknown date'">
                {{ page.metadata.date || 'Unknown' }}
              </p>
              <p class="text-xs font-medium truncate">
                {{ page.detection_count }} detections
              </p>
            </div>
          </div>
        </div>

        <!-- Loading State -->
        <div v-if="loading" class="text-center py-12">
          <div class="inline-block w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p class="text-muted-foreground mt-2">Loading pages...</p>
        </div>

        <!-- Empty State -->
        <div v-else-if="pages.length === 0" class="text-center py-12">
          <p class="text-muted-foreground">No pages found</p>
        </div>
      </div>

      <!-- Footer with Pagination -->
      <div v-if="totalPages > 1" class="flex items-center justify-center gap-4 p-4 border-t">
        <button
          @click="$emit('page-change', currentPage - 1)"
          :disabled="currentPage === 1"
          class="px-4 py-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          ← Previous
        </button>
        <span class="text-sm font-medium min-w-[100px] text-center">
          Page {{ currentPage }} of {{ totalPages }}
        </span>
        <button
          @click="$emit('page-change', currentPage + 1)"
          :disabled="currentPage >= totalPages"
          class="px-4 py-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-medium"
        >
          Next →
        </button>
      </div>
    </div>
  </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { drawAnnotations } from '@/lib/imageAnnotation'

export interface PageData {
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

interface Props {
  open: boolean
  pages: PageData[]
  loading?: boolean
  currentPage: number
  totalPages: number
  totalItems: number
  source: string
}

const props = defineProps<Props>()

defineEmits<{
  close: []
  select: [page: PageData]
  'page-change': [page: number]
}>()

const gridRef = ref<HTMLDivElement>()

async function renderPageThumbnail(page: PageData) {
  const canvas = document.getElementById(`canvas-${page.page_id}`) as HTMLCanvasElement
  if (!canvas) return

  const image = new Image()
  image.crossOrigin = 'anonymous'

  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve()
    image.onerror = () => reject(new Error('Failed to load image'))
    image.src = `/static/${props.source}/images/${page.image_path}`
  })

  // Draw with detections overlays
  drawAnnotations(canvas, image, page.detections, {
    maxWidth: 300,
    lineWidth: 1,
    fontSize: 8,
    showLabels: false,
  })
}

// Watch for page changes and render thumbnails
watch(() => props.pages, async (newPages) => {
  if (newPages.length > 0) {
    // Wait for next tick to ensure canvas refs are set
    await new Promise(resolve => setTimeout(resolve, 0))
    for (const page of newPages) {
      try {
        await renderPageThumbnail(page)
      } catch (error) {
        console.error('Failed to render thumbnail:', error)
      }
    }
  }
}, { immediate: true, deep: true })
</script>
