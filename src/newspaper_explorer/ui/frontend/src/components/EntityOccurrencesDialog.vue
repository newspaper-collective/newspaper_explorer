<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/lib/api'

interface EntityOccurrence {
  line_id: string
  source_id: string
  issue_id: string
  page_id: string
  text_block_id: string
  entity_text: string
  entity_type: string
  confidence: number
  detection_count: number
}

interface Props {
  open: boolean
  source: string
  entityText: string
  entityType: string
  runId?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  runId: null,
})

const emit = defineEmits<{
  'update:open': [value: boolean]
}>()

const router = useRouter()
const occurrences = ref<EntityOccurrence[]>([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = 20

const totalPages = computed(() => Math.ceil(occurrences.value.length / pageSize))
const paginatedOccurrences = computed(() => {
  const start = (currentPage.value - 1) * pageSize
  return occurrences.value.slice(start, start + pageSize)
})

async function loadOccurrences() {
  if (!props.entityText || !props.open) return

  loading.value = true
  try {
    const params: any = {
      entity_text: props.entityText,
      entity_type: props.entityType,
      limit: 500,
    }
    if (props.runId) params.run_id = props.runId

    const response = await api.get(`/entities/${props.source}/occurrences`, {
      params,
    })
    occurrences.value = response.data
    currentPage.value = 1
  } catch (error) {
    console.error('Failed to load entity occurrences:', error)
    occurrences.value = []
  } finally {
    loading.value = false
  }
}

function close() {
  emit('update:open', false)
}

function goToPage(occurrence: EntityOccurrence) {
  // Extract page number from page_id (last part)
  const pageNumber = parseInt(extractPageNumber(occurrence.page_id), 10)

  // Navigate to the issue viewer with the page number
  router.push({
    name: 'issue',
    params: { issueId: occurrence.issue_id },
    query: { page: pageNumber.toString() },
  })
  close()
}

function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`
}

function extractPageNumber(pageId: string): string {
  // Extract page number from page_id (e.g., "3074409-X_1901-01-08_006_1_002" -> "002")
  const parts = pageId.split('_')
  const lastPart = parts[parts.length - 1]
  return lastPart
}

function formatPageNumber(pageId: string): string {
  // Format page number without leading zeros for display
  return parseInt(extractPageNumber(pageId), 10).toString()
}

function nextPage() {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
  }
}

function prevPage() {
  if (currentPage.value > 1) {
    currentPage.value--
  }
}

watch(() => props.open, (isOpen) => {
  if (isOpen) {
    loadOccurrences()
  }
}, { immediate: true })
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[100] flex items-center justify-center bg-overlay-medium"
      @click.self="close"
    >
      <div class="relative w-full max-w-4xl max-h-[90vh] bg-background rounded-lg shadow-lg flex flex-col">
        <!-- Header -->
        <div class="flex items-start justify-between p-6 border-b">
          <div>
            <h2 class="text-2xl font-bold">{{ entityText }}</h2>
            <p class="text-sm text-muted-foreground mt-1">
              {{ entityType }} • {{ occurrences.length }} occurrence{{ occurrences.length !== 1 ? 's' : '' }}
            </p>
          </div>
          <button
            @click="close"
            class="rounded-md p-2 hover:bg-accent transition-colors"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              stroke-width="2"
              stroke-linecap="round"
              stroke-linejoin="round"
            >
              <path d="M18 6 6 18" />
              <path d="m6 6 12 12" />
            </svg>
          </button>
        </div>

        <!-- Content -->
        <div class="flex-1 overflow-y-auto p-6">
          <div v-if="loading" class="text-center py-12">
            <p class="text-muted-foreground">Loading occurrences...</p>
          </div>

          <div v-else-if="occurrences.length === 0" class="text-center py-12">
            <p class="text-muted-foreground">No occurrences found</p>
          </div>

          <div v-else class="space-y-3">
            <div
              v-for="occurrence in paginatedOccurrences"
              :key="occurrence.line_id"
              class="rounded-lg border bg-card p-4 hover:bg-accent transition-colors cursor-pointer"
              @click="goToPage(occurrence)"
            >
              <div class="flex items-start justify-between gap-4">
                <div class="flex-1">
                  <div class="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                    <span>Issue: {{ occurrence.issue_id }}</span>
                    <span>•</span>
                    <span>Page {{ formatPageNumber(occurrence.page_id) }}</span>
                    <span>•</span>
                    <span>Confidence: {{ formatConfidence(occurrence.confidence) }}</span>
                  </div>
                  <div class="text-xs font-mono text-muted-foreground mt-1">
                    {{ occurrence.text_block_id }}
                  </div>
                </div>
                <div class="flex-shrink-0">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class="text-muted-foreground"
                  >
                    <path d="m9 18 6-6-6-6" />
                  </svg>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Footer with pagination -->
        <div
          v-if="occurrences.length > pageSize"
          class="flex items-center justify-between p-4 border-t"
        >
          <p class="text-sm text-muted-foreground">
            Showing {{ (currentPage - 1) * pageSize + 1 }} -
            {{ Math.min(currentPage * pageSize, occurrences.length) }} of
            {{ occurrences.length }}
          </p>
          <div class="flex gap-2">
            <button
              @click="prevPage"
              :disabled="currentPage === 1"
              class="px-3 py-2 rounded-md border bg-background hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <button
              @click="nextPage"
              :disabled="currentPage === totalPages"
              class="px-3 py-2 rounded-md border bg-background hover:bg-accent disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>
