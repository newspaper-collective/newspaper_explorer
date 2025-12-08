<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-[100] flex items-center justify-center bg-black/80" @click="$emit('close')">
    <div class="relative w-[90vw] h-[90vh] bg-card rounded-lg shadow-xl flex flex-col" @click.stop>
      <!-- Header -->
      <div class="flex items-center justify-between p-4 border-b">
        <div>
          <h2 class="text-xl font-semibold">Picture Gallery</h2>
          <p class="text-sm text-muted-foreground">
            {{ totalItems.toLocaleString() }} pictures total • Page {{ currentPage }} of {{ totalPages }}
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
            v-for="picture in pictures"
            :key="picture.detection_id"
            :data-detection-id="picture.detection_id"
            @click="$emit('select', picture)"
            class="group cursor-pointer rounded-lg border bg-card hover:border-primary transition-all overflow-hidden"
          >
            <div class="aspect-square relative">
              <!-- Cropped Image -->
              <img
                v-if="croppedImages[picture.detection_id]"
                :src="croppedImages[picture.detection_id]"
                :alt="`Picture from ${picture.date || 'unknown date'}`"
                class="w-full h-full object-cover"
              />
              <!-- Loading State -->
              <div v-else class="w-full h-full flex items-center justify-center bg-muted">
                <div class="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
              </div>

              <!-- Hover Overlay -->
              <div class="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <span class="text-white text-sm font-medium">View Details</span>
              </div>
            </div>

            <!-- Info -->
            <div class="p-2 border-t">
              <p class="text-xs text-muted-foreground truncate" :title="picture.date || 'Unknown date'">
                {{ picture.date ? formatDate(picture.date) : 'Unknown' }}
              </p>
              <p class="text-xs font-medium truncate">
                {{ Math.round(picture.confidence * 100) }}% conf
              </p>
            </div>
          </div>
        </div>

        <!-- Loading State -->
        <div v-if="loading" class="text-center py-12">
          <div class="inline-block w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          <p class="text-muted-foreground mt-2">Loading pictures...</p>
        </div>

        <!-- Empty State -->
        <div v-else-if="pictures.length === 0" class="text-center py-12">
          <p class="text-muted-foreground">No pictures found</p>
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
import { onMounted, onUpdated, ref } from 'vue'
import { formatDate } from '@/lib/composables/useImageUtils'

export interface Picture {
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
  text_content?: string
  caption_id?: string
  caption_text?: string
  caption_bbox?: any
  [key: string]: any
}

interface Props {
  open: boolean
  pictures: Picture[]
  croppedImages: Record<string, string>
  loading?: boolean
  currentPage: number
  totalPages: number
  totalItems: number
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: []
  select: [picture: Picture]
  'page-change': [page: number]
  'load-image': [picture: Picture]
}>()

const gridRef = ref<HTMLDivElement>()

// Set up intersection observer to load images as they come into view
onMounted(() => {
  setupIntersectionObserver()
})

onUpdated(() => {
  setupIntersectionObserver()
})

function setupIntersectionObserver() {
  if (!gridRef.value) return

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const detectionId = entry.target.getAttribute('data-detection-id')
          const picture = props.pictures.find(p => p.detection_id === detectionId)
          if (picture) {
            emit('load-image', picture)
            observer.unobserve(entry.target)
          }
        }
      })
    },
    { root: gridRef.value, rootMargin: '100px' }
  )

  const cards = gridRef.value.querySelectorAll('[data-detection-id]')
  cards.forEach(card => observer.observe(card))
}
</script>

<!-- Close Teleport tag at end of template -->
