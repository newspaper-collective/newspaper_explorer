import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/lib/api'
import type { SourceInfo, SourceStats } from '@/types'

export const useSourceStore = defineStore('source', () => {
  const sources = ref<string[]>([])
  const currentSource = ref<string | null>(null)
  const sourceInfo = ref<SourceInfo | null>(null)
  const sourceStats = ref<SourceStats | null>(null)
  const startDate = ref<string | null>(null)
  const endDate = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  const hasCurrentSource = computed(() => currentSource.value !== null)

  async function loadSources() {
    loading.value = true
    error.value = null
    try {
      const response = await api.get<string[]>('/sources/')
      sources.value = response.data
      
      // Auto-select first source
      if (sources.value.length > 0 && !currentSource.value) {
        await selectSource(sources.value[0])
      }
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  async function selectSource(source: string) {
    currentSource.value = source
    loading.value = true
    error.value = null
    
    try {
      // Load source info
      const infoResponse = await api.get<SourceInfo>(`/sources/${source}`)
      sourceInfo.value = infoResponse.data

      // Load source stats
      const statsResponse = await api.get<SourceStats>(`/sources/${source}/stats`)
      sourceStats.value = statsResponse.data
      
      // Initialize dates from stats if available
      if (sourceStats.value?.date_range) {
        startDate.value = sourceStats.value.date_range[0]
        endDate.value = sourceStats.value.date_range[1]
      }
    } catch (e: any) {
      error.value = e.message
    } finally {
      loading.value = false
    }
  }

  return {
    sources,
    currentSource,
    sourceInfo,
    sourceStats,
    startDate,
    endDate,
    loading,
    error,
    hasCurrentSource,
    loadSources,
    selectSource,
  }
})
