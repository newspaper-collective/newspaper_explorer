<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import api from '@/lib/api'

interface AnalysisMetadata {
  source: string
  analysis_type: string
  run_id: string
  display_name: string
  row_count: number
  created_at?: string
  duration_seconds?: number
  parameters: Record<string, any>
  metadata: Record<string, any>
}

interface AnalysisRunInfo {
  run_id: string
  display_name: string
  source: string
  analysis_type: string
  created_at?: string
  duration_seconds?: number
  row_count: number
  parameters: Record<string, any>
}

interface Props {
  source: string
  analysisType: string
  runId?: string | null
}

const props = withDefaults(defineProps<Props>(), {
  runId: null,
})

const emit = defineEmits<{
  'update:runId': [value: string]
  'loaded': [metadata: AnalysisMetadata]
}>()

const metadata = ref<AnalysisMetadata | null>(null)
const availableRuns = ref<AnalysisRunInfo[]>([])
const loading = ref(false)
const error = ref<string | null>(null)
const selectedRunId = ref<string | null>(props.runId)

// Computed properties
const formattedDate = computed(() => {
  if (!metadata.value?.created_at) return 'Unknown date'
  try {
    return new Date(metadata.value.created_at).toLocaleString()
  } catch {
    return metadata.value.created_at
  }
})

const formattedDuration = computed(() => {
  if (!metadata.value?.duration_seconds) return 'Unknown'
  const seconds = metadata.value.duration_seconds
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`
  return `${(seconds / 3600).toFixed(1)}h`
})

// Load available runs
async function loadRuns() {
  try {
    const response = await api.get(
      `/results/${props.source}/${props.analysisType}/runs`
    )
    availableRuns.value = response.data

    // If no run selected and runs available, select the first (most recent)
    if (!selectedRunId.value && availableRuns.value.length > 0) {
      selectedRunId.value = availableRuns.value[0].run_id
    }
  } catch (err: any) {
    console.error('Failed to load runs:', err)
    error.value = err.response?.data?.detail || 'Failed to load analysis runs'
  }
}

// Load metadata for selected run
async function loadMetadata() {
  if (!selectedRunId.value) return

  loading.value = true
  error.value = null

  try {
    const response = await api.get(
      `/results/${props.source}/${props.analysisType}/metadata`,
      {
        params: { run_id: selectedRunId.value },
      }
    )
    metadata.value = response.data
    emit('loaded', response.data)
  } catch (err: any) {
    console.error('Failed to load metadata:', err)
    error.value = err.response?.data?.detail || 'Failed to load analysis metadata'
    metadata.value = null
  } finally {
    loading.value = false
  }
}

// Watch for run selection changes
watch(selectedRunId, (newRunId) => {
  if (newRunId) {
    emit('update:runId', newRunId)
    loadMetadata()
  }
})

// Watch for prop changes
watch(
  () => [props.source, props.analysisType],
  () => {
    selectedRunId.value = null
    metadata.value = null
    loadRuns()
  },
  { immediate: true }
)

// Expose methods for parent components
defineExpose({
  reload: loadMetadata,
  metadata,
})
</script>

<template>
  <div>
    <!-- Run selector card -->
    <div class="rounded-lg border bg-card p-3">
      <div class="flex items-end justify-between gap-4">
        <div class="flex-1 space-y-2">
          <label class="text-sm font-medium">Analysis Run</label>
          <select
            v-model="selectedRunId"
            class="w-full px-3 py-2 text-sm"
            :disabled="loading || availableRuns.length === 0"
          >
            <option v-if="availableRuns.length === 0" :value="null">
              No analysis runs available
            </option>
            <option
              v-for="run in availableRuns"
              :key="run.run_id"
              :value="run.run_id"
            >
              {{ run.display_name }}
            </option>
          </select>
        </div>

        <!-- Info icon with hover tooltip -->
        <div v-if="metadata" class="relative group mb-2">
          <div class="flex items-center justify-center text-muted-foreground cursor-help">
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
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4" />
              <path d="M12 8h.01" />
            </svg>
          </div>

          <!-- Hover tooltip -->
          <div class="absolute right-0 top-8 w-80 p-4 bg-popover border rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
            <h3 class="font-semibold mb-3">Analysis Details</h3>

            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-muted-foreground">Records:</span>
                <span class="font-medium">{{ metadata.row_count.toLocaleString() }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-muted-foreground">Created:</span>
                <span class="font-medium">{{ formattedDate }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-muted-foreground">Duration:</span>
                <span class="font-medium">{{ formattedDuration }}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-muted-foreground">Run ID:</span>
                <span class="font-mono text-xs break-all">{{ metadata.run_id }}</span>
              </div>
            </div>

            <!-- Parameters -->
            <div v-if="Object.keys(metadata.parameters).length > 0" class="mt-3 pt-3 border-t">
              <h4 class="font-medium mb-2">Parameters</h4>
              <div class="space-y-1 text-sm">
                <div
                  v-for="(value, key) in metadata.parameters"
                  :key="key"
                  class="flex justify-between gap-2"
                >
                  <span class="text-muted-foreground">{{ key }}:</span>
                  <span class="font-mono text-xs break-all text-right">
                    {{ Array.isArray(value) ? value.join(', ') : value }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Quick stats -->
      <div v-if="metadata" class="mt-3 text-xs text-muted-foreground">
        <span>{{ metadata.row_count.toLocaleString() }} records</span>
        <span class="mx-1">•</span>
        <span>{{ formattedDate }}</span>
        <span class="mx-1">•</span>
        <span>{{ formattedDuration }}</span>
      </div>
    </div>

    <!-- Error display -->
    <div
      v-if="error"
      class="rounded-lg border border-destructive bg-destructive/10 p-4 text-sm text-destructive"
    >
      {{ error }}
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="text-center py-8 text-muted-foreground">
      Loading analysis data...
    </div>

    <!-- Content slot -->
    <slot v-else :metadata="metadata" :loading="loading" />
  </div>
</template>
