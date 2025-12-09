<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import {
  Calendar,
  FileText,
  Image as ImageIcon,
  FolderOpen,
  BookOpen,
  CalendarDays,
} from 'lucide-vue-next'

const sourceStore = useSourceStore()

const stats = computed(() => sourceStore.sourceStats)
const info = computed(() => sourceStore.sourceInfo)

// Helper to format numbers
const formatNumber = (num: number) => new Intl.NumberFormat().format(num)

// Calculate years span
const yearsSpan = computed(() => {
  if (!stats.value?.date_range) return 0
  const [start, end] = stats.value.date_range
  const startYear = new Date(start).getFullYear()
  const endYear = new Date(end).getFullYear()
  return endYear - startYear + 1
})

// Random lines
const randomLines = ref<any[]>([])
const loadingLines = ref(false)

const fetchRandomLines = async () => {
  if (!sourceStore.currentSource) return

  loadingLines.value = true
  try {
    const response = await fetch(`/api/data/${sourceStore.currentSource}/random-lines?count=5`)
    if (response.ok) {
      randomLines.value = await response.json()
    }
  } catch (error) {
    console.error('Failed to fetch random lines:', error)
  } finally {
    loadingLines.value = false
  }
}

// Random images
const randomImages = ref<any[]>([])
const loadingImages = ref(false)

const fetchRandomImages = async () => {
  if (!sourceStore.currentSource) return

  loadingImages.value = true
  try {
    const response = await fetch(`/api/data/${sourceStore.currentSource}/random-images?count=5`)
    if (response.ok) {
      randomImages.value = await response.json()
    }
  } catch (error) {
    console.error('Failed to fetch random images:', error)
  } finally {
    loadingImages.value = false
  }
}

onMounted(() => {
  fetchRandomLines()
  fetchRandomImages()
})

const handleSourceChange = (event: Event) => {
  const target = event.target as HTMLSelectElement
  sourceStore.selectSource(target.value)
}

const formatSource = (source: string) => {
  return source
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}
</script>

<template>
  <div class="space-y-6 px-4">
    <!-- Header -->
    <!-- Source Selector -->
    <div class="max-w-xl mx-auto mt-8">
      <div class="rounded-lg border bg-card text-card-foreground shadow-sm p-4 flex items-center gap-4">
        <h3 class="text-sm font-medium text-muted-foreground shrink-0">Collection</h3>
        <select
          v-model="sourceStore.currentSource"
          @change="handleSourceChange"
          class="flex-1 h-9 px-3 py-1 text-sm"
        >
          <option v-for="source in sourceStore.sources" :key="source" :value="source">
            {{ formatSource(source) }}
          </option>
        </select>
        <span class="text-xs text-muted-foreground shrink-0">{{ sourceStore.sources.length }} available</span>
      </div>
    </div>

    <!-- Compact Stats Grid - 6 cards in 2 rows -->
    <div v-if="stats" class="grid gap-3 grid-cols-3">
      <!-- Text Lines -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <FileText class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Text Lines</p>
            <p class="text-xl font-bold">{{ formatNumber(stats.total_lines) }}</p>
            <p v-if="stats.total_blocks > 0" class="text-xs text-muted-foreground">
              {{ formatNumber(stats.total_blocks) }} text blocks
            </p>
          </div>
        </div>
      </div>

      <!-- Date Range -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <CalendarDays class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Date Range</p>
            <p class="text-xl font-bold">{{ yearsSpan }} years</p>
            <p class="text-xs text-muted-foreground truncate">
              {{ stats.date_range[0] }} – {{ stats.date_range[1] }}
            </p>
          </div>
        </div>
      </div>

      <!-- Source Files -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <FolderOpen class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Source Files</p>
            <p class="text-xl font-bold">{{ formatNumber(stats.total_issues) }}</p>
            <p class="text-xs text-muted-foreground">ALTO XML format</p>
          </div>
        </div>
      </div>

      <!-- Total Issues -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <Calendar class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Issues</p>
            <p class="text-xl font-bold">{{ formatNumber(stats.total_issues) }}</p>
            <p class="text-xs text-muted-foreground">newspaper editions</p>
          </div>
        </div>
      </div>

      <!-- Average Pages -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <BookOpen class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Avg Pages/Issue</p>
            <p class="text-xl font-bold">{{ (stats.total_pages / stats.total_issues).toFixed(1) }}</p>
            <p class="text-xs text-muted-foreground">
              {{ formatNumber(stats.total_pages) }} pages total
            </p>
          </div>
        </div>
      </div>

      <!-- Images -->
      <div class="rounded-lg border bg-card p-4">
        <div class="flex items-center gap-3">
          <ImageIcon class="h-6 w-6 text-muted-foreground shrink-0" />
          <div class="min-w-0">
            <p class="text-xs font-medium text-muted-foreground">Images</p>
            <p class="text-xl font-bold">{{ info?.has_images ? '✓' : '—' }}</p>
            <p v-if="info?.image_size" class="text-xs text-muted-foreground">
              {{ info.image_size }} ({{ formatNumber(info.image_count || 0) }} images)
            </p>
            <p v-else class="text-xs text-muted-foreground">
              {{ info?.has_images ? 'Available' : 'Not available' }}
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- Random Samples Grid -->
    <div class="grid gap-6 grid-cols-2">
      <!-- Random Lines Sample -->
      <div class="rounded-lg border bg-card">
        <div class="p-4 border-b">
          <div class="flex items-center justify-between">
            <h2 class="text-xl font-bold">Random Text Sample</h2>
            <button
              @click="fetchRandomLines"
              :disabled="loadingLines"
              class="px-3 py-1 text-sm border rounded hover:bg-accent disabled:opacity-50"
            >
              {{ loadingLines ? 'Loading...' : 'Refresh' }}
            </button>
          </div>
        </div>

        <div class="p-6">
          <div v-if="randomLines.length > 0" class="space-y-3">
            <div
              v-for="(line, idx) in randomLines"
              :key="line.line_id"
              class="p-3 rounded bg-muted/50 border"
            >
              <div class="flex items-start gap-3">
                <span class="text-xs font-mono text-muted-foreground shrink-0">{{ idx + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <p class="text-sm">{{ line.text }}</p>
                  <p class="text-xs text-muted-foreground mt-1">
                    {{ new Date(line.date).toLocaleDateString() }} • Page {{ line.page_number }}
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div v-else-if="loadingLines" class="text-center text-muted-foreground py-8">
            Loading random lines...
          </div>

          <div v-else class="text-center text-muted-foreground py-8">
            No data available
          </div>
        </div>
      </div>

      <!-- Random Images Sample -->
      <div class="rounded-lg border bg-card">
        <div class="p-4 border-b">
          <div class="flex items-center justify-between">
            <h2 class="text-xl font-bold">Random Image Sample</h2>
            <button
              @click="fetchRandomImages"
              :disabled="loadingImages"
              class="px-3 py-1 text-sm border rounded hover:bg-accent disabled:opacity-50"
            >
              {{ loadingImages ? 'Loading...' : 'Refresh' }}
            </button>
          </div>
        </div>

        <div class="p-6">
          <div v-if="randomImages.length > 0" class="flex gap-3 overflow-x-auto pb-2">
            <div
              v-for="(image, idx) in randomImages"
              :key="image.url"
              class="flex-shrink-0 w-48"
            >
              <div class="p-2 rounded bg-muted/50 border h-full flex flex-col">
                <div class="flex-1">
                  <span class="text-xs font-mono text-muted-foreground">{{ idx + 1 }}</span>
                  <img
                    :src="image.url"
                    :alt="`Random image ${idx + 1}`"
                    class="w-full h-auto rounded border mt-1"
                    loading="lazy"
                  />
                </div>
                <div class="mt-2 text-xs text-muted-foreground space-y-1">
                  <p v-if="image.date">{{ new Date(image.date).toLocaleDateString() }}</p>
                  <p v-if="image.page_number">Page {{ image.page_number }}</p>
                </div>
              </div>
            </div>
          </div>

          <div v-else-if="loadingImages" class="text-center text-muted-foreground py-8">
            Loading random images...
          </div>

          <div v-else class="text-center text-muted-foreground py-8">
            No images available
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
