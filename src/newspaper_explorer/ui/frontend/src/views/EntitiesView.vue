<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, PieChart, LineChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import api from '@/lib/api'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Input } from '@/components/ui/input'
import { Slider } from '@/components/ui/slider'
import ResultsViewer from '@/components/ResultsViewer.vue'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import EntityOccurrencesDialog from '@/components/EntityOccurrencesDialog.vue'
import { getEntityColor, getEntityTypeColors, ENTITY_TYPES } from '@/lib/colors'
import {
  useBarChart,
  usePieChart,
  useWordCloud,
  useTimelineChart,
} from '@/lib/charts'
import type { EChartsOption } from 'echarts'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  PieChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
])

interface Entity {
  entity_text: string
  entity_type: string
  detection_count: number
  avg_confidence: number
}

const sourceStore = useSourceStore()
const entities = ref<Entity[]>([])
const loading = ref(false)
const selectedRunId = ref<string | null>(null)
const selectedType = ref<string | null>(null)
const entityTypes = ref<string[]>([])
const resultsViewer = ref<InstanceType<typeof ResultsViewer>>()
const searchQuery = ref('')
const minConfidence = ref(0)
const wordcloudEntityCount = ref(100)

// Slider array adapters (Slider component uses arrays)
const minConfidenceSlider = computed({
  get: () => [minConfidence.value],
  set: (val: number[]) => { minConfidence.value = val[0] }
})
const wordcloudEntityCountSlider = computed({
  get: () => [wordcloudEntityCount.value],
  set: (val: number[]) => { wordcloudEntityCount.value = val[0] }
})

// Chart options
const topEntitiesChart = ref<EChartsOption>({})
const typeDistributionChart = ref<EChartsOption>({})
const wordcloudChart = ref<EChartsOption>({})
const timelineChart = ref<EChartsOption>({})

// Dialog state
const dialogOpen = ref(false)
const selectedEntity = ref<Entity | null>(null)

// Computed filtered entities
const filteredEntities = computed(() => {
  return entities.value.filter(e => {
    // Filter by confidence
    if (e.avg_confidence < minConfidence.value / 100) return false

    // Filter by search query
    if (searchQuery.value && !e.entity_text.toLowerCase().includes(searchQuery.value.toLowerCase())) {
      return false
    }

    return true
  })
})

async function loadEntityTypes() {
  if (!sourceStore.currentSource) return

  try {
    const params: any = {}
    if (selectedRunId.value) params.run_id = selectedRunId.value

    const response = await api.get(
      `/entities/${sourceStore.currentSource}/types`,
      { params }
    )
    entityTypes.value = response.data
  } catch (error) {
    console.error('Failed to load entity types:', error)
  }
}

async function loadEntities() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    const params: any = {} // No limit - get all entities
    if (selectedType.value) params.entity_type = selectedType.value
    if (selectedRunId.value) params.run_id = selectedRunId.value

    const response = await api.get(
      `/entities/${sourceStore.currentSource}/`,
      { params }
    )
    entities.value = response.data
    updateCharts()
    await loadTimeline()
  } catch (error) {
    console.error('Failed to load entities:', error)
    entities.value = []
  } finally {
    loading.value = false
  }
}

async function loadTimeline() {
  if (!sourceStore.currentSource) return

  try {
    const params: any = { aggregation: 'month' }
    if (selectedType.value) params.entity_type = selectedType.value
    if (selectedRunId.value) params.run_id = selectedRunId.value

    const response = await api.get(
      `/entities/${sourceStore.currentSource}/timeline`,
      { params }
    )
    createTimelineChart(response.data)
  } catch (error) {
    console.error('Failed to load timeline:', error)
  }
}

function getEntityTypeColor(entityType: string): string {
  return getEntityColor(entityType)
}

function capitalizeEntityType(entityType: string): string {
  return entityType.toUpperCase()
}

function updateCharts() {
  // Top entities bar chart with colored bars by type
  const topEntities = filteredEntities.value.slice(0, 20)
  const entityTypeMap = new Map(topEntities.map(e => [e.entity_text, e.entity_type]))

  const baseChart = useBarChart(
    topEntities.map((e) => ({
      name: e.entity_text,
      value: e.detection_count,
    })),
    {
      title: {
        text: 'Top Entities by Frequency',
        left: 'center',
        textStyle: {
          fontSize: 16,
          fontWeight: 'normal',
        },
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow',
        },
        formatter: (params: any) => {
          const data = params[0]
          const entityType = entityTypeMap.get(data.name) || 'unknown'
          return `${data.name} (${capitalizeEntityType(entityType)})<br/>Occurrences: <strong>${data.value}</strong>`
        },
      },
    }
  )

  // Add color-coding to series data
  if (baseChart.series && Array.isArray(baseChart.series) && baseChart.series[0]) {
    const series = baseChart.series[0] as any
    series.data = topEntities.map((e) => ({
      name: e.entity_text,
      value: e.detection_count,
      itemStyle: { color: getEntityTypeColor(e.entity_type) }
    }))
  }

  topEntitiesChart.value = baseChart

  // Type distribution pie chart with consistent colors
  const typeCount = new Map<string, number>()
  filteredEntities.value.forEach((e) => {
    typeCount.set(e.entity_type, (typeCount.get(e.entity_type) || 0) + e.detection_count)
  })

  const typeData = Array.from(typeCount.entries()).map(([name, value]) => ({
    name,
    value,
    itemStyle: { color: getEntityTypeColor(name) }
  }))

  typeDistributionChart.value = usePieChart(typeData, {
    title: {
      text: 'Entity Type Distribution',
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 'normal',
      },
    },
    color: Object.values(getEntityTypeColors()),
  })

  // Wordcloud with entity type colors
  const wordcloudData = filteredEntities.value.slice(0, wordcloudEntityCount.value).map((e) => ({
    name: e.entity_text,
    value: e.detection_count,
    textStyle: { color: getEntityTypeColor(e.entity_type) }
  }))

  wordcloudChart.value = useWordCloud(wordcloudData, {
    title: {
      text: `Entity Word Cloud (Top ${wordcloudData.length})`,
      left: 'center',
      textStyle: {
        fontSize: 16,
        fontWeight: 'normal',
      },
    },
  })
}

function createTimelineChart(data: Record<string, { date: string; value: number }[]>) {
  if (!data || Object.keys(data).length === 0) {
    timelineChart.value = {}
    return
  }

  // Create color array in the same order as the series will be created (Object.keys order)
  const entityTypes = Object.keys(data)
  const colors = entityTypes.map(type => getEntityTypeColor(type))

  // Pass colors via a custom option that useTimelineChart will apply to series
  timelineChart.value = useTimelineChart(data, 'bar', {
    title: {
      text: 'Entity Mentions Over Time',
      left: 'center',
      textStyle: { fontSize: 16, fontWeight: 'normal' },
    },
    // Don't pass color at top-level - it causes issues with replaceMerge
    // Colors are applied in useTimelineChart via itemStyle
    _colors: colors,
  })
}

function openEntityDialog(entity: Entity) {
  selectedEntity.value = entity
  dialogOpen.value = true
}

// Watch for filter changes
watch([selectedType, searchQuery, minConfidence, wordcloudEntityCount], () => {
  if (entities.value.length > 0) {
    updateCharts()
  }
})

// Watch for type filter changes (requires reload)
watch(selectedType, () => {
  loadEntities()
})

// Watch for run changes
watch(selectedRunId, () => {
  loadEntityTypes()
  loadEntities()
})

function onMetadataLoaded() {
  loadEntityTypes()
  loadEntities()
  loadTimeline()
}

onMounted(() => {
  if (sourceStore.currentSource) {
    loadEntityTypes()
  }
})
</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex flex-wrap items-start gap-4">
        <!-- Title -->
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Named Entities"
            description="Explore named entities extracted from the newspaper articles."
            icon="entities"
          />
        </div>

        <!-- Results Selector -->
        <div class="flex-1 min-w-[200px] self-stretch">
          <ResultsViewer
            v-if="sourceStore.currentSource"
            ref="resultsViewer"
            :source="sourceStore.currentSource"
            analysis-type="entities"
            v-model:run-id="selectedRunId"
            @loaded="onMetadataLoaded"
          />
        </div>

        <!-- Inline Filters Card -->
        <div class="rounded-lg border bg-card p-3 flex items-center self-stretch">
          <div class="flex flex-col gap-2 w-full">
            <!-- Search -->
            <Input
              v-model="searchQuery"
              type="text"
              placeholder="Search..."
              class="h-8 text-sm"
            />

            <!-- Type Filter -->
            <Select v-model="selectedType">
              <SelectTrigger class="w-full text-sm">
                <SelectValue placeholder="All Types" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem :value="null">All Types</SelectItem>
                <SelectItem v-for="type in entityTypes" :key="type" :value="type">
                  {{ capitalizeEntityType(type) }}
                </SelectItem>
              </SelectContent>
            </Select>

            <!-- Confidence and Count Row -->
            <div class="flex items-center justify-between gap-3">
              <!-- Confidence Filter -->
              <div class="flex items-center gap-2 flex-1">
                <label class="text-xs text-muted-foreground whitespace-nowrap">
                  Conf: {{ minConfidence }}%
                </label>
                <Slider
                  v-model="minConfidenceSlider"
                  :min="0"
                  :max="100"
                  :step="5"
                  class="flex-1"
                />
              </div>

              <!-- Results count -->
              <span class="text-xs text-muted-foreground whitespace-nowrap">
                {{ filteredEntities.length }}/{{ entities.length }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 pb-6 space-y-6">
      <!-- Timeline Chart -->
      <div v-if="entities.length > 0 && timelineChart.series" class="rounded-lg border bg-card p-4">
        <VChart :option="timelineChart" class="h-[400px]" autoresize />
      </div>

      <!-- Charts -->
      <div v-if="entities.length > 0" class="grid gap-6 lg:grid-cols-2">
        <!-- Top Entities Chart -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="topEntitiesChart" class="h-[400px]" autoresize />
        </div>

        <!-- Type Distribution Chart -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="typeDistributionChart" class="h-[400px]" autoresize />
        </div>
      </div>

      <!-- Wordcloud -->
      <div v-if="entities.length > 0" class="rounded-lg border bg-card p-4">
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <label class="text-xs text-muted-foreground">Size: {{ wordcloudEntityCount }}</label>
            <Slider
              v-model="wordcloudEntityCountSlider"
              :min="20"
              :max="200"
              :step="10"
              class="w-32"
            />
          </div>
        </div>
        <VChart :option="wordcloudChart" class="h-[600px]" autoresize />
      </div>

      <!-- Loading state -->
      <div v-if="loading" class="text-center py-12">
        <p class="text-muted-foreground">Loading entities...</p>
      </div>

      <!-- Entity cards grid -->
      <div v-else-if="filteredEntities.length > 0" class="rounded-lg border bg-card p-6">
        <h3 class="text-lg font-semibold mb-4">Entity Details</h3>
        <div class="grid gap-3 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <div
            v-for="entity in filteredEntities"
            :key="`${entity.entity_text}-${entity.entity_type}`"
            class="rounded-md border bg-background p-3 hover:bg-accent transition-colors cursor-pointer"
            @click="openEntityDialog(entity)"
          >
            <div class="flex items-start justify-between gap-2">
              <h4 class="font-semibold text-sm">{{ entity.entity_text }}</h4>
              <span
                class="text-[10px] text-white px-1.5 py-0.5 rounded whitespace-nowrap font-medium"
                :style="{ backgroundColor: getEntityTypeColor(entity.entity_type) }"
              >
                {{ capitalizeEntityType(entity.entity_type) }}
              </span>
            </div>
            <div class="mt-2 flex items-center gap-4 text-xs text-muted-foreground">
              <span>{{ entity.detection_count }} occurrences</span>
              <span>{{ (entity.avg_confidence * 100).toFixed(1) }}% conf.</span>
            </div>
          </div>
        </div>
      </div>

      <!-- No data state -->
      <div v-else-if="!loading" class="text-center py-12 text-muted-foreground">
        <p>No entities found{{ selectedType ? ' for this type' : '' }}</p>
      </div>

      <!-- Entity Occurrences Dialog -->
      <EntityOccurrencesDialog
        v-if="selectedEntity && sourceStore.currentSource"
        v-model:open="dialogOpen"
        :source="sourceStore.currentSource"
        :entity-text="selectedEntity.entity_text"
        :entity-type="selectedEntity.entity_type"
        :run-id="selectedRunId"
      />
    </div>
  </div>
</template>
