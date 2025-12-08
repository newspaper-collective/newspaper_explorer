<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useSourceStore } from '@/stores/source'
import { useRouter } from 'vue-router'
import api from '@/lib/api'
import OpenSeadragonViewer from '@/components/OpenSeadragonViewer.vue'
import { ImageOff, Home, Eye, X, ChevronLeft, ChevronRight } from 'lucide-vue-next'

interface Props {
  issueId: string
  showBackButton?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showBackButton: true,
})

const emit = defineEmits<{
  back: []
  selectPage: [pageNumber: number]
}>()

const sourceStore = useSourceStore()
const router = useRouter()
const loading = ref(false)

// Issue metadata
const metadata = ref<any>(null)
const pages = ref<any[]>([])

// Preview dialog state
const previewOpen = ref(false)
const previewPageIndex = ref(0)

const previewPage = computed(() => {
  if (previewPageIndex.value >= 0 && previewPageIndex.value < pages.value.length) {
    return pages.value[previewPageIndex.value]
  }
  return null
})

const previewImageUrl = computed(() => {
  if (!previewPage.value || !sourceStore.currentSource) return ''
  // Use the image_url directly from the page data (already contains full path)
  if (previewPage.value.image_url) {
    return previewPage.value.image_url
  }
  return ''
})

function openPreview(index: number, event: Event) {
  event.stopPropagation()
  previewPageIndex.value = index
  previewOpen.value = true
}

function closePreview() {
  previewOpen.value = false
}

function previousPreviewPage() {
  if (previewPageIndex.value > 0) {
    previewPageIndex.value--
  }
}

function nextPreviewPage() {
  if (previewPageIndex.value < pages.value.length - 1) {
    previewPageIndex.value++
  }
}

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

async function loadIssue() {
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
    }
  } catch (error) {
    console.error('Failed to load issue:', error)
  } finally {
    loading.value = false
  }
}

function openPage(pageNumber: number) {
  // Navigate to issue reader with page parameter
  router.push({
    name: 'issue',
    params: { issueId: props.issueId },
    query: { page: pageNumber.toString() }
  })
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

function goToMonth() {
  if (metadata.value?.year && metadata.value?.month) {
    router.push({
      name: 'browse',
      query: { year: metadata.value.year.toString(), month: metadata.value.month.toString() }
    })
  } else {
    emit('back')
  }
}

function formatDate(dateString: string): string {
  const date = new Date(dateString)
  return date.toLocaleDateString('de-DE')
}

onMounted(() => {
  loadIssue()
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
          @click="goToMonth"
          class="text-lg font-medium text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
        >
          {{ monthName }}
        </button>
        <span class="text-muted-foreground text-lg">›</span>
        <span class="text-lg font-medium text-foreground">
          {{ formatDate(metadata.date) }}
        </span>
      </template>
      </div>
    </div>

    <!-- Issue Metadata Subtitle -->
    <div v-if="metadata" class="text-sm text-muted-foreground">
      {{ formatDate(metadata.date) }}
      <template v-if="metadata.issueNumber">
        • Issue {{ metadata.issueNumber }}
      </template>
      <template v-if="metadata.dailyCount">
        ({{ metadata.dailyCount }} of day)
      </template>
      • {{ metadata.pageCount }} {{ metadata.pageCount === 1 ? 'page' : 'pages' }}
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="text-center py-12">
      <p class="text-muted-foreground">Loading pages...</p>
    </div>

    <!-- Page Grid -->
    <div v-else-if="pages.length > 0">
      <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
        <button
          v-for="(page, index) in pages"
          :key="page.page_id"
          @click="openPage(page.page_number)"
          class="group relative rounded-lg border bg-card overflow-hidden hover:border-primary transition-all hover:shadow-lg"
        >
          <!-- Thumbnail -->
          <div class="aspect-[3/4] bg-muted relative max-h-[calc(100vh-24rem)]">
            <img
              v-if="page.has_image && page.image_url"
              :src="page.image_url"
              :alt="`Page ${page.page_number}`"
              class="w-full h-full object-cover"
              loading="lazy"
            />
            <div
              v-else
              class="w-full h-full flex items-center justify-center text-muted-foreground"
            >
              <ImageOff class="h-12 w-12" />
            </div>

            <!-- Overlay on hover -->
            <div class="absolute inset-0 bg-transparent group-hover:bg-overlay-subtle transition-colors" />
          </div>

          <!-- Page Number Label with Preview Icon -->
          <div class="p-3 flex items-center justify-center gap-2">
            <button
              v-if="page.has_image"
              @click="openPreview(index, $event)"
              class="p-1 rounded hover:bg-accent transition-colors text-muted-foreground hover:text-foreground"
              title="Preview page"
            >
              <Eye class="h-4 w-4" />
            </button>
            <p class="text-sm font-medium">Page {{ page.page_number }}</p>
          </div>
        </button>
      </div>
    </div>

    <!-- No Data -->
    <div v-else class="rounded-lg border bg-card p-8 text-center">
      <p class="text-muted-foreground">No pages found</p>
    </div>

    <!-- Preview Dialog -->
    <div
      v-if="previewOpen"
      class="fixed inset-0 z-50 flex items-center justify-center bg-overlay-heavy"
      @click="closePreview"
    >
      <div
        class="bg-card rounded-lg shadow-lg w-[70vw] max-w-[1000px] h-[95vh] overflow-hidden m-4 flex flex-col"
        @click.stop
      >
        <!-- Dialog Header -->
        <div class="flex items-center justify-between p-4 border-b bg-muted/50">
          <div v-if="previewPage" class="space-y-1">
            <p class="font-semibold text-base">
              Page {{ previewPage.page_number }} of {{ pages.length }}
            </p>
            <p class="text-xs text-muted-foreground">
              {{ metadata?.date ? formatDate(metadata.date) : '' }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <!-- Navigation Controls -->
            <button
              @click="previousPreviewPage"
              :disabled="previewPageIndex === 0"
              class="p-2 hover:bg-accent rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Previous page"
            >
              <ChevronLeft class="w-5 h-5" />
            </button>
            <span class="text-sm text-muted-foreground min-w-[60px] text-center">
              {{ previewPageIndex + 1 }} / {{ pages.length }}
            </span>
            <button
              @click="nextPreviewPage"
              :disabled="previewPageIndex === pages.length - 1"
              class="p-2 hover:bg-accent rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Next page"
            >
              <ChevronRight class="w-5 h-5" />
            </button>
            <div class="w-px h-6 bg-border mx-2" />
            <button
              @click="closePreview"
              class="p-2 hover:bg-accent rounded-md transition-colors"
              title="Close preview"
            >
              <X class="w-5 h-5" />
            </button>
          </div>
        </div>
        <!-- OpenSeadragon Viewer -->
        <div class="flex-1 relative">
          <OpenSeadragonViewer
            v-if="previewImageUrl"
            :key="previewPageIndex"
            :image-url="previewImageUrl"
            :current-page="previewPageIndex + 1"
            :total-pages="pages.length"
            @change-page="(delta) => { if (delta < 0) previousPreviewPage(); else nextPreviewPage(); }"
          />
        </div>
      </div>
    </div>
  </div>
</template>
