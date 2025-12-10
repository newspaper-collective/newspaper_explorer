<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useSourceStore } from '@/stores/source'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart, LineChart, ScatterChart, BoxplotChart } from 'echarts/charts'
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
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import ResultsViewer from '@/components/ResultsViewer.vue'
import { getEmotionColor, getChartAxisPointerColor, EMOTIONS as EMOTION_LIST } from '@/lib/colors'
import type { EChartsOption } from 'echarts'

// Register ECharts components
use([
  CanvasRenderer,
  BarChart,
  LineChart,
  ScatterChart,
  BoxplotChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  ToolboxComponent,
  DataZoomComponent,
])

const sourceStore = useSourceStore()
const loading = ref(false)
const selectedRunId = ref<string | null>(null)

// Data state
const statistics = ref<any>(null)
const timelineYear = ref<any[]>([])
const timelineMonth = ref<any[]>([])
const peaks = ref<any[]>([])

// UI state
const selectedPeakEmotion = ref('sadness')
const EMOTIONS = [...EMOTION_LIST] // Use centralized list

// Chart Options
const distributionChart = ref<EChartsOption>({})
const timelineChart = ref<EChartsOption>({})
const streamChart = ref<EChartsOption>({})
const eraChart = ref<EChartsOption>({})
const peaksChart = ref<EChartsOption>({})

async function loadData() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    // Load statistics
    const statsRes = await api.get(`/emotions/${sourceStore.currentSource}/statistics`)
    statistics.value = statsRes.data

    // Load timelines
    const yearRes = await api.get(`/emotions/${sourceStore.currentSource}/timeline`, {
      params: { granularity: 'year' }
    })
    timelineYear.value = yearRes.data

    const monthRes = await api.get(`/emotions/${sourceStore.currentSource}/timeline`, {
      params: { granularity: 'month' }
    })
    timelineMonth.value = monthRes.data

    updateCharts()
    await loadPeaks()
  } catch (error) {
    console.error('Failed to load emotion data:', error)
  } finally {
    loading.value = false
  }
}

async function loadPeaks() {
  if (!sourceStore.currentSource) return

  try {
    const res = await api.get(`/emotions/${sourceStore.currentSource}/peaks`, {
      params: { emotion: selectedPeakEmotion.value, limit: 20 }
    })
    peaks.value = res.data
    updatePeaksChart()
  } catch (error) {
    console.error('Failed to load peaks:', error)
  }
}

function updateCharts() {
  if (!statistics.value) return

  // 1. Overall Distribution
  const means = statistics.value.overall_means
  const total = Object.values(means).reduce((a: any, b: any) => a + b, 0) as number

  distributionChart.value = {
    title: { text: 'Overall Emotion Distribution', left: 'center' },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' }, formatter: '{b}: {c}%' },
    grid: { bottom: 30 },
    xAxis: {
      type: 'category',
      data: Object.keys(means).map(k => k.charAt(0).toUpperCase() + k.slice(1))
    },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' } },
    series: [{
      data: Object.entries(means).map(([k, v]: [string, any]) => ({
        value: Number((v / total * 100).toFixed(1)),
        itemStyle: { color: getEmotionColor(k) }
      })),
      type: 'bar'
    }]
  }

  // 2. Timeline (Line Chart)
  if (timelineYear.value.length > 0) {
    timelineChart.value = {
      title: { text: 'Emotion Trends Over Years', left: 'center' },
      tooltip: { trigger: 'axis' },
      legend: { bottom: 0 },
      grid: { bottom: 80 },
      xAxis: { type: 'category', data: timelineYear.value.map(d => d.time_key) },
      yAxis: { type: 'value' },
      series: EMOTIONS.map(emo => ({
        name: emo.charAt(0).toUpperCase() + emo.slice(1),
        type: 'line',
        data: timelineYear.value.map(d => d[emo]),
        itemStyle: { color: getEmotionColor(emo) },
        smooth: true
      }))
    }
  }

  // 3. Stream Chart (Stacked Area)
  if (timelineMonth.value.length > 0) {
    // Sort order: negative to positive
    const plotOrder = ['agitation', 'anger', 'fear', 'sadness', 'love', 'joy']

    streamChart.value = {
      title: { text: 'Relative Emotion Shares (Monthly)', left: 'center' },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross', label: { backgroundColor: getChartAxisPointerColor() } } },
      legend: { bottom: 0 },
      grid: { bottom: 80, left: '3%', right: '4%', containLabel: true },
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: timelineMonth.value.map(d => d.time_key)
      },
      yAxis: { type: 'value', max: 100 },
      series: plotOrder.map(emo => ({
        name: emo.charAt(0).toUpperCase() + emo.slice(1),
        type: 'line',
        stack: 'Total',
        areaStyle: {},
        emphasis: { focus: 'series' },
        data: timelineMonth.value.map(d => {
          // Calculate share for this month
          const rowTotal = EMOTIONS.reduce((sum, e) => sum + (d[e] || 0), 0)
          return rowTotal > 0 ? Number(((d[emo] || 0) / rowTotal * 100).toFixed(1)) : 0
        }),
        itemStyle: { color: getEmotionColor(emo) },
        showSymbol: false
      }))
    }
  }

  // 4. Era Analysis (Box Plots)
  const eraStats = statistics.value.era_statistics
  if (eraStats) {
    const eras = ['pre_war', 'war', 'post_war']
    const datasetSource: any[] = []

    // Prepare data for boxplot
    // We need to construct series for each emotion
    const series: any[] = []

    EMOTIONS.forEach(emo => {
      const data = eras.map(era => {
        const s = eraStats[era]?.[emo]
        if (!s) return []
        // ECharts boxplot expects [min, Q1, median, Q3, max]
        return [s.min, s.q1, s.median, s.q3, s.max]
      })

      series.push({
        name: emo.charAt(0).toUpperCase() + emo.slice(1),
        type: 'boxplot',
        data: data,
        itemStyle: { color: getEmotionColor(emo) }
      })
    })

    eraChart.value = {
      title: { text: 'Emotion Distribution by Era', left: 'center' },
      legend: { bottom: 0 },
      tooltip: { trigger: 'item', axisPointer: { type: 'shadow' } },
      grid: { bottom: 80 },
      xAxis: {
        type: 'category',
        data: eras.map(e => e.replace('_', ' ').toUpperCase())
      },
      yAxis: { type: 'value' },
      series: series
    }
  }
}

function updatePeaksChart() {
  if (peaks.value.length === 0) return

  peaksChart.value = {
    title: {
      text: `Top ${selectedPeakEmotion.value} Peaks`,
      left: 'center'
    },
    tooltip: {
      formatter: (params: any) => {
        const data = params.data
        return `${data[0]}<br/>Value: ${data[1].toFixed(3)}<br/>Era: ${data[2]}`
      }
    },
    xAxis: { type: 'category', splitLine: { show: false } },
    yAxis: { type: 'value', splitLine: { show: false } },
    series: [{
      name: selectedPeakEmotion.value,
      type: 'scatter',
      symbolSize: (val: any) => Math.max(val[1] * 20, 10),
      data: peaks.value.map(p => {
        const emotionCapitalized = selectedPeakEmotion.value.charAt(0).toUpperCase() + selectedPeakEmotion.value.slice(1)
        return [p.date, p[`${emotionCapitalized}_prob`], p.era]
      }),
      itemStyle: { color: getEmotionColor(selectedPeakEmotion.value) }
    }]
  }
}

watch(selectedPeakEmotion, () => {
  loadPeaks()
})

onMounted(() => {
  if (sourceStore.currentSource) {
    loadData()
  }
})
</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex flex-wrap items-start gap-4">
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Emotions"
            description="Analyze emotional content and trends over time."
            icon="emotions"
          />
        </div>

        <div class="flex-1 min-w-[200px] self-stretch">
          <ResultsViewer
            v-if="sourceStore.currentSource"
            :source="sourceStore.currentSource"
            analysis-type="emotions"
            v-model:run-id="selectedRunId"
            @loaded="loadData"
          />
        </div>
      </div>
    </div>

    <!-- Content -->
    <div class="px-4 pb-6 space-y-6">
      <div v-if="loading" class="text-center py-12">
        <p class="text-muted-foreground">Loading emotion analysis...</p>
      </div>

      <template v-else-if="statistics">
        <!-- Top Row: Distribution & Timeline -->
        <div class="grid gap-6 md:grid-cols-2">
          <div class="rounded-lg border bg-card p-4">
            <VChart :option="distributionChart" class="h-[400px]" autoresize />
          </div>
          <div class="rounded-lg border bg-card p-4">
            <VChart :option="timelineChart" class="h-[400px]" autoresize />
          </div>
        </div>

        <!-- Stream Chart -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="streamChart" class="h-[500px]" autoresize />
        </div>

        <!-- Era Analysis -->
        <div class="rounded-lg border bg-card p-4">
          <VChart :option="eraChart" class="h-[400px]" autoresize />
        </div>

        <!-- Peaks Analysis -->
        <div class="rounded-lg border bg-card p-6">
          <div class="flex items-center justify-between mb-6">
            <h3 class="text-lg font-semibold">Notable Emotion Peaks</h3>
            <div class="flex items-center gap-2">
              <span class="text-sm text-muted-foreground">Select Emotion:</span>
              <Select v-model="selectedPeakEmotion">
                <SelectTrigger class="w-[140px] text-sm">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem v-for="emo in EMOTIONS" :key="emo" :value="emo">
                    {{ emo.charAt(0).toUpperCase() + emo.slice(1) }}
                  </SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div class="grid gap-6 md:grid-cols-2">
            <!-- Peaks Table -->
            <div class="overflow-hidden rounded-md border">
              <table class="w-full text-sm">
                <thead class="bg-muted/50">
                  <tr>
                    <th class="p-3 text-left font-medium">Date</th>
                    <th class="p-3 text-left font-medium">Era</th>
                    <th class="p-3 text-right font-medium">Intensity</th>
                  </tr>
                </thead>
                <tbody class="divide-y">
                  <tr v-for="peak in peaks" :key="peak.line_id" class="hover:bg-muted/50">
                    <td class="p-3">{{ peak.date }}</td>
                    <td class="p-3 capitalize">{{ peak.era?.replace('_', ' ') || '-' }}</td>
                    <td class="p-3 text-right font-mono">
                      {{ peak[`${selectedPeakEmotion.charAt(0).toUpperCase() + selectedPeakEmotion.slice(1)}_prob`]?.toFixed(3) || 'N/A' }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Peaks Scatter -->
            <div class="h-[400px]">
              <VChart :option="peaksChart" class="h-full" autoresize />
            </div>
          </div>
        </div>

        <!-- Summary Stats Table -->
        <div class="rounded-lg border bg-card p-6">
          <h3 class="text-lg font-semibold mb-4">Statistics by Era</h3>
          <div class="grid gap-6 md:grid-cols-3">
            <div v-for="era in ['pre_war', 'war', 'post_war']" :key="era" class="space-y-3">
              <h4 class="font-medium text-center uppercase text-muted-foreground tracking-wider text-xs">
                {{ era.replace('_', ' ') }}
              </h4>
              <div class="rounded-md border divide-y">
                <div
                  v-for="emo in EMOTIONS"
                  :key="emo"
                  class="flex items-center justify-between p-2 text-sm"
                >
                  <span class="capitalize">{{ emo }}</span>
                  <span class="font-mono text-muted-foreground">
                    {{ statistics.era_statistics[era]?.[emo]?.mean.toFixed(3) || '-' }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </template>

      <div v-else class="text-center py-12 text-muted-foreground">
        No emotion data available.
      </div>
    </div>
  </div>
</template>
