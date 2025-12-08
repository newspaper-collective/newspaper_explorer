<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import api from '@/lib/api'
import {
  Database,
  FileText,
  Image,
  CheckCircle,
  XCircle,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  Loader2,
  FolderOpen,
  Archive,
  Play,
  AlertCircle,
  Users,
  MessageSquare,
  Tag,
  Network,
  Smile,
  LayoutDashboard,
  ExternalLink,
  HardDrive,
  FileCode,
  FileArchive,
} from 'lucide-vue-next'

const sourceStore = useSourceStore()

// Types
interface SourceOverview {
  name: string
  title: string
  language: string
  years: string
  totalLines: number
  totalIssues: number
  totalBlocks: number
  imageCount: number
  imageSize: string | null
  xmlFileCount: number | null
  xmlTotalSize: string | null
  parquetSize: string | null
  archiveSize: string | null
  hasText: boolean
  hasParsedData: boolean
  hasImages: boolean
  analysisResults: Record<string, { count: number }>
}

interface PreprocessedDataset {
  name: string
  path: string
  created: string
  steps: number
  source: string
  row_count?: number
}

interface AnalysisRun {
  runId: string
  displayName: string
  source: string
  analysisType: string
  createdAt: string
  entityCount?: number
  rowCount?: number
}

interface AnalysisResult {
  source: string
  analysisType: string
  runCount: number
  runs: AnalysisRun[]
}

// Local state
const loading = ref(false)
const allSources = ref<SourceOverview[]>([])
const allPreprocessedDatasets = ref<PreprocessedDataset[]>([])
const allAnalysisResults = ref<AnalysisResult[]>([])

// Section collapse state
const expandedSections = ref<Record<string, boolean>>({
  sources: true,
  preprocessed: true,
  results: true,
})

// Analysis type icons
const analysisIcons: Record<string, any> = {
  entities: Users,
  emotions: Smile,
  topics: MessageSquare,
  keywords: Tag,
  concepts: Network,
  layout: LayoutDashboard,
}

// Fetch all data
async function fetchAllData() {
  loading.value = true
  try {
    await Promise.all([
      fetchAllSources(),
      fetchAllPreprocessedDatasets(),
      fetchAllAnalysisResults(),
    ])
  } finally {
    loading.value = false
  }
}

// Fetch source overviews
async function fetchAllSources() {
  const sources: SourceOverview[] = []

  for (const sourceName of sourceStore.sources) {
    try {
      const [infoRes, statsRes] = await Promise.all([
        api.get(`/sources/${sourceName}`),
        api.get(`/sources/${sourceName}/stats`).catch(() => ({ data: null })),
      ])

      const info = infoRes.data
      const stats = statsRes.data

      sources.push({
        name: sourceName,
        title: info.metadata?.newspaper_title || sourceName,
        language: info.metadata?.language || 'unknown',
        years: info.metadata?.years_available || '',
        totalLines: stats?.total_lines || 0,
        totalIssues: stats?.total_issues || 0,
        totalBlocks: stats?.total_blocks || 0,
        imageCount: info.image_count || 0,
        imageSize: info.image_size || null,
        xmlFileCount: info.xml_file_count || null,
        xmlTotalSize: info.xml_total_size || null,
        parquetSize: info.parquet_size || null,
        archiveSize: info.total_archive_size || null,
        hasText: info.has_text || false,
        hasParsedData: (stats?.total_lines || 0) > 0,
        hasImages: info.has_images || false,
        analysisResults: info.analysis_results || {},
      })
    } catch (error) {
      console.error(`Failed to fetch info for ${sourceName}:`, error)
    }
  }

  allSources.value = sources
}

// Fetch all preprocessed datasets across all sources
async function fetchAllPreprocessedDatasets() {
  const datasets: PreprocessedDataset[] = []

  for (const sourceName of sourceStore.sources) {
    try {
      const response = await api.get(`/preprocessing/${sourceName}/datasets`)
      for (const dataset of response.data) {
        datasets.push({
          ...dataset,
          source: sourceName,
        })
      }
    } catch {
      // Source might not have preprocessed data
    }
  }

  // Sort by creation date, newest first
  allPreprocessedDatasets.value = datasets.sort((a, b) => b.created.localeCompare(a.created))
}

// Fetch all analysis results across all sources
async function fetchAllAnalysisResults() {
  const results: AnalysisResult[] = []

  for (const sourceName of sourceStore.sources) {
    try {
      // Get list of analysis types for this source
      const typesResponse = await api.get(`/results/${sourceName}/available-analyses`)

      for (const analysis of typesResponse.data) {
        // Fetch all runs for this analysis type
        const runsResponse = await api.get(`/results/${sourceName}/${analysis.analysis_type}/runs`)
        const runs: AnalysisRun[] = runsResponse.data.map((run: any) => ({
          runId: run.run_id,
          displayName: run.display_name,
          source: sourceName,
          analysisType: analysis.analysis_type,
          createdAt: run.created_at,
          entityCount: run.entity_count,
          rowCount: run.row_count,
        }))

        results.push({
          source: sourceName,
          analysisType: analysis.analysis_type,
          runCount: runs.length,
          runs,
        })
      }
    } catch {
      // Source might not have analysis results
    }
  }

  allAnalysisResults.value = results
}

// Initialize
onMounted(async () => {
  if (sourceStore.sources.length === 0) {
    await sourceStore.loadSources()
  }
  await fetchAllData()
})

// Toggle section expansion
function toggleSection(section: string) {
  expandedSections.value[section] = !expandedSections.value[section]
}

// Format number with locale
function formatNumber(num: number): string {
  return new Intl.NumberFormat().format(num)
}

// Helper to parse size strings (e.g., "1.5 GB", "350 MB") to GB
function parseSizeToGB(sizeStr: string | null): number {
  if (!sizeStr) return 0
  const gbMatch = sizeStr.match(/^([\d.]+)\s*GB/i)
  if (gbMatch) return parseFloat(gbMatch[1])
  const mbMatch = sizeStr.match(/^([\d.]+)\s*MB/i)
  if (mbMatch) return parseFloat(mbMatch[1]) / 1024
  return 0
}

// Helper to format GB to appropriate unit
function formatSize(sizeGB: number): string | null {
  if (sizeGB <= 0) return null
  if (sizeGB >= 1.0) return `${sizeGB.toFixed(1)} GB`
  return `${(sizeGB * 1024).toFixed(0)} MB`
}

// Summary stats
const totalStats = computed(() => {
  // Aggregate sizes
  const totalImageSizeGB = allSources.value.reduce((sum, s) => sum + parseSizeToGB(s.imageSize), 0)
  const totalXmlSizeGB = allSources.value.reduce((sum, s) => sum + parseSizeToGB(s.xmlTotalSize), 0)
  const totalParquetSizeGB = allSources.value.reduce((sum, s) => sum + parseSizeToGB(s.parquetSize), 0)
  const totalArchiveSizeGB = allSources.value.reduce((sum, s) => sum + parseSizeToGB(s.archiveSize), 0)
  const totalXmlFiles = allSources.value.reduce((sum, s) => sum + (s.xmlFileCount || 0), 0)

  return {
    sources: allSources.value.length,
    sourcesWithData: allSources.value.filter(s => s.hasParsedData).length,
    totalLines: allSources.value.reduce((sum, s) => sum + s.totalLines, 0),
    totalIssues: allSources.value.reduce((sum, s) => sum + s.totalIssues, 0),
    totalImages: allSources.value.reduce((sum, s) => sum + s.imageCount, 0),
    totalImageSize: formatSize(totalImageSizeGB),
    totalXmlFiles,
    totalXmlSize: formatSize(totalXmlSizeGB),
    totalParquetSize: formatSize(totalParquetSizeGB),
    totalArchiveSize: formatSize(totalArchiveSizeGB),
    preprocessedDatasets: allPreprocessedDatasets.value.length,
    analysisResults: allAnalysisResults.value.length,
  }
})

// Group results by source
const resultsBySource = computed(() => {
  const grouped: Record<string, AnalysisResult[]> = {}
  for (const result of allAnalysisResults.value) {
    if (!grouped[result.source]) {
      grouped[result.source] = []
    }
    grouped[result.source].push(result)
  }
  return grouped
})

// Total run count across all sources and analysis types
const totalRunCount = computed(() =>
  allAnalysisResults.value.reduce((sum, r) => sum + r.runCount, 0)
)

</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex items-center justify-between">
        <AnalysisHeader
          title="Data Overview"
          description="View all source datasets, preprocessed data, and analysis results"
        />

        <div class="flex items-center gap-3">
          <button
            @click="fetchAllData"
            class="h-9 px-3 rounded-md border border-input bg-background text-sm shadow-sm hover:bg-accent flex items-center gap-2"
            :disabled="loading"
          >
            <RefreshCw :class="['h-4 w-4', loading && 'animate-spin']" />
            Refresh
          </button>
        </div>
      </div>
    </div>

    <!-- Content -->
    <div class="px-4 pb-6 space-y-6">
      <!-- Loading state -->
      <div v-if="loading && allSources.length === 0" class="flex items-center justify-center py-12">
        <Loader2 class="h-8 w-8 animate-spin text-muted-foreground" />
      </div>

      <template v-else>
        <!-- Summary Cards -->
        <div class="flex flex-wrap gap-3">
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <Database class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ totalStats.sourcesWithData }}/{{ totalStats.sources }}</div>
              <div class="text-xs text-muted-foreground">Sources</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <FileCode class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ formatNumber(totalStats.totalXmlFiles) }}</div>
              <div class="text-xs text-muted-foreground">
                XML Files
                <span v-if="totalStats.totalXmlSize" class="ml-1">({{ totalStats.totalXmlSize }})</span>
              </div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <HardDrive class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ totalStats.totalParquetSize || '—' }}</div>
              <div class="text-xs text-muted-foreground">Parquet Data</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <FileText class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ formatNumber(totalStats.totalLines) }}</div>
              <div class="text-xs text-muted-foreground">Text Lines</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <FolderOpen class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ formatNumber(totalStats.totalIssues) }}</div>
              <div class="text-xs text-muted-foreground">Issues</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <Image class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ formatNumber(totalStats.totalImages) }}</div>
              <div class="text-xs text-muted-foreground">
                Images
                <span v-if="totalStats.totalImageSize" class="ml-1">({{ totalStats.totalImageSize }})</span>
              </div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <FileArchive class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ totalStats.totalArchiveSize || '—' }}</div>
              <div class="text-xs text-muted-foreground">Archives</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <Play class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ totalStats.preprocessedDatasets }}</div>
              <div class="text-xs text-muted-foreground">Preprocessed</div>
            </div>
          </div>
          <div class="rounded-lg border bg-card px-4 py-3 flex items-center gap-3">
            <Archive class="h-5 w-5 text-muted-foreground shrink-0" />
            <div>
              <div class="text-lg font-bold">{{ totalStats.analysisResults }}</div>
              <div class="text-xs text-muted-foreground">Analyses</div>
            </div>
          </div>
        </div>

        <!-- Source Datasets Section -->
        <div class="rounded-lg border bg-card">
          <button
            @click="toggleSection('sources')"
            class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
          >
            <div class="flex items-center gap-3">
              <Database class="h-5 w-5 text-muted-foreground" />
              <div>
                <h3 class="font-medium">Source Datasets</h3>
                <p class="text-sm text-muted-foreground">Raw XML data and parsed parquet files</p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-sm text-muted-foreground">{{ allSources.length }} sources</span>
              <component :is="expandedSections.sources ? ChevronDown : ChevronRight" class="h-4 w-4 text-muted-foreground" />
            </div>
          </button>

          <div v-if="expandedSections.sources" class="border-t">
            <div class="overflow-x-auto">
              <table class="w-full">
                <thead class="bg-muted/50">
                  <tr>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Source</th>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Years</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">XML Files</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">Parquet</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">Issues</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">Lines</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">Images</th>
                    <th class="text-center text-xs font-medium text-muted-foreground px-4 py-3">Status</th>
                  </tr>
                </thead>
                <tbody class="divide-y">
                  <tr
                    v-for="source in allSources"
                    :key="source.name"
                    class="hover:bg-muted/30 transition-colors"
                  >
                    <td class="px-4 py-3">
                      <div class="font-medium">{{ source.title }}</div>
                      <div class="text-xs text-muted-foreground">{{ source.name }}</div>
                    </td>
                    <td class="px-4 py-3 text-sm">{{ source.years }}</td>
                    <td class="px-4 py-3 text-sm text-right font-mono">
                      <div v-if="source.xmlFileCount">
                        {{ formatNumber(source.xmlFileCount) }}
                        <div v-if="source.xmlTotalSize" class="text-xs text-muted-foreground">{{ source.xmlTotalSize }}</div>
                      </div>
                      <span v-else class="text-muted-foreground">—</span>
                    </td>
                    <td class="px-4 py-3 text-sm text-right font-mono">
                      <span v-if="source.parquetSize">{{ source.parquetSize }}</span>
                      <span v-else class="text-muted-foreground">—</span>
                    </td>
                    <td class="px-4 py-3 text-sm text-right font-mono">{{ formatNumber(source.totalIssues) }}</td>
                    <td class="px-4 py-3 text-sm text-right font-mono">{{ formatNumber(source.totalLines) }}</td>
                    <td class="px-4 py-3 text-sm text-right font-mono">
                      <div v-if="source.hasImages">
                        {{ formatNumber(source.imageCount) }}
                        <div v-if="source.imageSize" class="text-xs text-muted-foreground">{{ source.imageSize }}</div>
                      </div>
                      <span v-else class="text-muted-foreground">—</span>
                    </td>
                    <td class="px-4 py-3 text-center">
                      <span
                        v-if="source.hasParsedData"
                        class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800"
                      >
                        <CheckCircle class="h-3 w-3" />
                        Ready
                      </span>
                      <span
                        v-else-if="source.hasText"
                        class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800"
                      >
                        <AlertCircle class="h-3 w-3" />
                        Raw Only
                      </span>
                      <span
                        v-else
                        class="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground"
                      >
                        <XCircle class="h-3 w-3" />
                        No Data
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Preprocessed Datasets Section -->
        <div class="rounded-lg border bg-card">
          <button
            @click="toggleSection('preprocessed')"
            class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
          >
            <div class="flex items-center gap-3">
              <Play class="h-5 w-5 text-muted-foreground" />
              <div>
                <h3 class="font-medium">Preprocessed Datasets</h3>
                <p class="text-sm text-muted-foreground">Results from preprocessing pipelines</p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-sm text-muted-foreground">{{ allPreprocessedDatasets.length }} datasets</span>
              <component :is="expandedSections.preprocessed ? ChevronDown : ChevronRight" class="h-4 w-4 text-muted-foreground" />
            </div>
          </button>

          <div v-if="expandedSections.preprocessed" class="border-t">
            <div v-if="allPreprocessedDatasets.length === 0" class="p-8 text-center text-muted-foreground">
              <Play class="h-10 w-10 mx-auto mb-3 opacity-50" />
              <p class="text-sm">No preprocessed datasets yet</p>
              <p class="text-xs mt-1">Create one in the Preprocessing tab</p>
              <router-link
                to="/preprocessing"
                class="inline-flex items-center gap-2 mt-4 px-4 py-2 rounded-md bg-primary text-primary-foreground text-sm hover:bg-primary/90"
              >
                <ExternalLink class="h-4 w-4" />
                Go to Preprocessing
              </router-link>
            </div>

            <div v-else class="overflow-x-auto">
              <table class="w-full">
                <thead class="bg-muted/50">
                  <tr>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Dataset Name</th>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Source</th>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Created</th>
                    <th class="text-right text-xs font-medium text-muted-foreground px-4 py-3">Steps</th>
                    <th class="text-left text-xs font-medium text-muted-foreground px-4 py-3">Path</th>
                  </tr>
                </thead>
                <tbody class="divide-y">
                  <tr
                    v-for="dataset in allPreprocessedDatasets"
                    :key="dataset.path"
                    class="hover:bg-muted/30 transition-colors"
                  >
                    <td class="px-4 py-3">
                      <div class="font-medium font-mono text-sm">{{ dataset.name }}</div>
                    </td>
                    <td class="px-4 py-3 text-sm">{{ dataset.source }}</td>
                    <td class="px-4 py-3 text-sm text-muted-foreground">{{ dataset.created }}</td>
                    <td class="px-4 py-3 text-sm text-right font-mono">{{ dataset.steps }}</td>
                    <td class="px-4 py-3 text-xs text-muted-foreground font-mono truncate max-w-[300px]" :title="dataset.path">
                      {{ dataset.path }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Analysis Results Section -->
        <div class="rounded-lg border bg-card">
          <button
            @click="toggleSection('results')"
            class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
          >
            <div class="flex items-center gap-3">
              <Archive class="h-5 w-5 text-muted-foreground" />
              <div>
                <h3 class="font-medium">Analysis Results</h3>
                <p class="text-sm text-muted-foreground">Entity extraction, emotions, topics, and more</p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <span class="text-sm text-muted-foreground">{{ totalRunCount }} runs</span>
              <component :is="expandedSections.results ? ChevronDown : ChevronRight" class="h-4 w-4 text-muted-foreground" />
            </div>
          </button>

          <div v-if="expandedSections.results" class="border-t">
            <div v-if="allAnalysisResults.length === 0" class="p-8 text-center text-muted-foreground">
              <Archive class="h-10 w-10 mx-auto mb-3 opacity-50" />
              <p class="text-sm">No analysis results yet</p>
              <p class="text-xs mt-1">Run analyses from the respective tabs (Entities, Emotions, etc.)</p>
            </div>

            <div v-else>
              <!-- Group by source -->
              <div v-for="(results, sourceName) in resultsBySource" :key="sourceName" class="border-b last:border-b-0">
                <div class="px-4 py-2 bg-muted/30">
                  <span class="text-sm font-medium">{{ sourceName }}</span>
                  <span class="text-xs text-muted-foreground ml-2">
                    {{ results.reduce((sum, r) => sum + r.runCount, 0) }} runs across {{ results.length }} analysis types
                  </span>
                </div>
                <!-- Analysis types for this source -->
                <div v-for="result in results" :key="`${result.source}-${result.analysisType}`" class="border-t first:border-t-0">
                  <!-- Analysis type header -->
                  <div class="flex items-center gap-3 px-4 py-2 bg-muted/10">
                    <component
                      :is="analysisIcons[result.analysisType] || Archive"
                      class="h-4 w-4 text-muted-foreground"
                    />
                    <span class="text-sm font-medium capitalize">{{ result.analysisType }}</span>
                    <span class="text-xs text-muted-foreground">{{ result.runCount }} {{ result.runCount === 1 ? 'run' : 'runs' }}</span>
                  </div>
                  <!-- Individual runs -->
                  <div class="divide-y">
                    <div
                      v-for="run in result.runs"
                      :key="run.runId"
                      class="flex items-center justify-between px-4 py-2 pl-11 hover:bg-muted/20"
                    >
                      <div>
                        <div class="text-sm">{{ run.displayName }}</div>
                        <div class="text-xs text-muted-foreground">
                          {{ new Date(run.createdAt).toLocaleDateString() }}
                          <span v-if="run.entityCount" class="ml-2">{{ formatNumber(run.entityCount) }} entities</span>
                          <span v-if="run.rowCount" class="ml-2">{{ formatNumber(run.rowCount) }} rows</span>
                        </div>
                      </div>
                      <router-link
                        :to="`/${result.analysisType}?run=${run.runId}`"
                        class="text-xs text-primary hover:underline flex items-center gap-1"
                      >
                        View
                        <ExternalLink class="h-3 w-3" />
                      </router-link>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </template>
    </div>
  </div>
</template>
