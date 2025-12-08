<script setup lang="ts">
import { ref, onMounted, watch, nextTick } from 'vue'
import { Info } from 'lucide-vue-next'
import { drawAnnotations, type Detection } from '@/lib/imageAnnotation'
import { useSourceStore } from '@/stores/source'

interface Props {
  pageId: string
  imagePath: string
  detections: Detection[]
  metadata?: {
    date?: string
    issue_number?: string
    daily_count?: string
    page_number?: string
  }
  maxWidth?: number
}

const props = withDefaults(defineProps<Props>(), {
  maxWidth: 400,
})

const emit = defineEmits<{
  viewDetails: []
  viewImage: []
}>()

const sourceStore = useSourceStore()
const canvas = ref<HTMLCanvasElement | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)

const weekdayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

function formatDate(dateStr: string): { formatted: string; weekday: string } {
  try {
    const date = new Date(dateStr)
    const weekday = weekdayNames[date.getDay()]
    const formatted = date.toLocaleDateString('en-GB', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    })
    return { formatted, weekday }
  } catch {
    return { formatted: dateStr, weekday: '' }
  }
}

async function loadAndAnnotateImage() {
  if (!canvas.value || !props.imagePath) return
  if (!sourceStore.currentSource) return

  loading.value = true
  error.value = null

  // Clear canvas before loading new image
  const ctx = canvas.value.getContext('2d')
  if (ctx) {
    ctx.clearRect(0, 0, canvas.value.width, canvas.value.height)
  }

  try {
    const image = new Image()
    image.crossOrigin = 'anonymous'

    await new Promise<void>((resolve, reject) => {
      image.onload = () => resolve()
      image.onerror = () => reject(new Error('Failed to load image'))
      // Add timestamp to prevent browser caching
      image.src = `/static/${sourceStore.currentSource}/images/${props.imagePath}?t=${Date.now()}`
    })

    drawAnnotations(canvas.value, image, props.detections, {
      maxWidth: props.maxWidth,
      lineWidth: 2,
      fontSize: 12,
      showLabels: true,
    })

    loading.value = false
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load image'
    loading.value = false
  }
}

onMounted(async () => {
  await nextTick()
  loadAndAnnotateImage()
})

watch(() => [props.imagePath, props.detections], async () => {
  await nextTick()
  loadAndAnnotateImage()
}, { deep: true })

const dateInfo = props.metadata?.date ? formatDate(props.metadata.date) : null
</script>

<template>
  <div class="rounded-lg border bg-card overflow-hidden">
    <!-- Header -->
    <div class="p-4 border-b bg-muted/50">
      <div v-if="metadata?.date" class="space-y-1">
        <p class="font-semibold text-base">
          {{ dateInfo?.formatted }} <span v-if="dateInfo?.weekday" class="text-muted-foreground">({{ dateInfo.weekday }})</span>
        </p>
        <p class="text-xs text-muted-foreground">
          Issue {{ metadata.issue_number }} • Daily {{ metadata.daily_count }} • Page {{ metadata.page_number }}
        </p>
      </div>
      <p v-else class="font-semibold text-sm">{{ pageId }}</p>
      <p class="text-sm text-muted-foreground mt-1">
        {{ detections.length }} detection{{ detections.length !== 1 ? 's' : '' }}
      </p>
    </div>

    <!-- Image Display -->
    <div 
      class="relative bg-black cursor-pointer hover:opacity-90 transition-opacity" 
      :style="{ height: '384px' }"
      @click="emit('viewImage')"
      title="Click to view full size"
    >
      <div v-if="loading" class="absolute inset-0 flex items-center justify-center z-10">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>

      <div v-if="error" class="absolute inset-0 flex items-center justify-center z-10">
        <p class="text-destructive text-center px-4">{{ error }}</p>
      </div>

      <canvas
        ref="canvas"
        class="w-full h-full object-contain"
        :class="{ 'opacity-0': loading || error }"
      />
    </div>

    <!-- Footer -->
    <div class="p-3 border-t flex justify-end">
      <button
        @click="emit('viewDetails')"
        class="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md hover:bg-accent"
      >
        <Info class="w-4 h-4" />
        View Details
      </button>
    </div>
  </div>
</template>
