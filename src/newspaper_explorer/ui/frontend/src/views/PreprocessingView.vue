<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useSourceStore } from '@/stores/source'
import { usePreprocessingStore } from '@/stores/preprocessing'
import { useDebounceFn } from '@vueuse/core'
import AnalysisHeader from '@/components/AnalysisHeader.vue'
import {
  Type,
  Sparkles,
  Eraser,
  Filter,
  Languages,
  CheckCircle,
  Settings,
  X,
  GripVertical,
  Play,
  RefreshCw,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  Loader2,
  Clock,
  Terminal,
  Copy,
  Database,
  Minus,
  FileText,
} from 'lucide-vue-next'
import type { PreprocessingStepInfo, PipelineStep } from '@/types/preprocessing'
import { CATEGORY_INFO } from '@/types/preprocessing'

const sourceStore = useSourceStore()
const preprocessingStore = usePreprocessingStore()

// Local state
const showStepConfig = ref(false)
const configStep = ref<PipelineStep | null>(null)
const showIntermediateSteps = ref(true)
const selectedPreset = ref<string | null>(null)
const showCLIExport = ref(false)
const draggedStepName = ref<string | null>(null)
const dragOverIndex = ref<number | null>(null)
const showInputSelector = ref(false)
const preprocessedDatasets = ref<Array<{name: string, path: string, created: string, steps: number, source: string}>>([])

// Section collapse state
const pipelineCollapsed = ref(false)
const previewCollapsed = ref(false)
const runSectionCollapsed = ref(false)

// Input selection state - can be either a source or a preprocessed dataset
const inputType = ref<'source' | 'dataset'>('source')
const selectedDataset = ref<{name: string, path: string, source: string} | null>(null)

// Scroll synchronization refs
const originalScrollContainer = ref<HTMLElement | null>(null)
const processedScrollContainer = ref<HTMLElement | null>(null)
const isScrolling = ref(false)

function onOriginalScroll(event: Event) {
  if (isScrolling.value) return
  const target = event.target as HTMLElement
  if (processedScrollContainer.value) {
    isScrolling.value = true
    processedScrollContainer.value.scrollTop = target.scrollTop
    requestAnimationFrame(() => { isScrolling.value = false })
  }
}

function onProcessedScroll(event: Event) {
  if (isScrolling.value) return
  const target = event.target as HTMLElement
  if (originalScrollContainer.value) {
    isScrolling.value = true
    originalScrollContainer.value.scrollTop = target.scrollTop
    requestAnimationFrame(() => { isScrolling.value = false })
  }
}

// Category icons map
const categoryIcons: Record<string, any> = {
  normalization: Type,
  modernization: Sparkles,
  cleaning: Eraser,
  filtering: Filter,
  linguistic: Languages,
  quality: CheckCircle,
}

// Initialize
onMounted(async () => {
  await Promise.all([
    preprocessingStore.loadSteps(),
    preprocessingStore.loadPresets(),
  ])

  // Load sample text if source is available
  if (sourceStore.currentSource) {
    await loadSamplesForCurrentPipeline()
  }
})

// Load samples based on whether dehyphenation is in pipeline
async function loadSamplesForCurrentPipeline() {
  if (!sourceStore.currentSource) return

  const hasDehyphenation = preprocessingStore.pipeline.some(s =>
    s.name.includes('dehyphenat')
  )

  await preprocessingStore.loadSampleText(
    sourceStore.currentSource,
    hasDehyphenation ? 'hyphenated' : 'random'
  )
}

// Load preprocessed datasets for all sources
async function loadAllPreprocessedDatasets() {
  const allDatasets: Array<{name: string, path: string, created: string, steps: number, source: string}> = []

  for (const source of sourceStore.sources) {
    try {
      const response = await fetch(`/api/preprocessing/${source}/datasets`)
      if (response.ok) {
        const datasets = await response.json()
        allDatasets.push(...datasets.map((d: any) => ({ ...d, source })))
      }
    } catch (error) {
      console.error(`Failed to load datasets for ${source}:`, error)
    }
  }

  // Sort by creation date, newest first
  preprocessedDatasets.value = allDatasets.sort((a, b) => b.created.localeCompare(a.created))
}

// Watch for input selector opening
watch(showInputSelector, async (isOpen) => {
  if (isOpen) {
    await loadAllPreprocessedDatasets()
  }
})

// Select a raw source as input
function selectSourceInput(sourceName: string) {
  inputType.value = 'source'
  selectedDataset.value = null
  sourceStore.selectSource(sourceName)
  showInputSelector.value = false
  loadSamplesForCurrentPipeline()
}

// Select a preprocessed dataset as input
function selectDatasetInput(dataset: {name: string, path: string, source: string}) {
  inputType.value = 'dataset'
  selectedDataset.value = dataset
  sourceStore.selectSource(dataset.source) // Set source for context
  showInputSelector.value = false
  loadSamplesForCurrentPipeline()
}

// Get display name for current input
const currentInputDisplay = computed(() => {
  if (inputType.value === 'dataset' && selectedDataset.value) {
    return `${selectedDataset.value.source} / ${selectedDataset.value.name}`
  }
  return sourceStore.currentSource || 'Select input...'
})

// Watch for source changes to reload sample text
watch(() => sourceStore.currentSource, async (newSource) => {
  if (newSource && !preprocessingStore.customTextMode) {
    await loadSamplesForCurrentPipeline()
  }
})

// Watch for pipeline changes to reload samples if dehyphenation added/removed
watch(() => preprocessingStore.pipeline, async () => {
  if (sourceStore.currentSource && !preprocessingStore.customTextMode) {
    const hasDehyphenation = preprocessingStore.pipeline.some(s =>
      s.name.includes('dehyphenat')
    )
    // Only reload if dehyphenation state changed
    const currentSampleType = preprocessingStore.previewSamples.some(s =>
      s.text?.endsWith('-')
    )
    if (hasDehyphenation !== currentSampleType) {
      await loadSamplesForCurrentPipeline()
    }
  }
  debouncedPreview()
}, { deep: true })

// Debounced preview update
const debouncedPreview = useDebounceFn(() => {
  preprocessingStore.preview(showIntermediateSteps.value)
  // Also run batch preview if we have samples
  if (preprocessingStore.previewSamples.length > 0) {
    preprocessingStore.previewBatch(showIntermediateSteps.value)
  }
}, 500)

watch(() => preprocessingStore.previewText, () => {
  debouncedPreview()
})

// Drag and drop handlers for step palette
function onDragStartFromPalette(event: DragEvent, stepName: string) {
  draggedStepName.value = stepName
  event.dataTransfer!.effectAllowed = 'copy'
  event.dataTransfer!.setData('text/plain', stepName)
}

function onDragOver(event: DragEvent, index: number) {
  event.preventDefault()
  dragOverIndex.value = index
}

function onDragLeave() {
  dragOverIndex.value = null
}

function onDrop(event: DragEvent, index: number) {
  event.preventDefault()

  const stepName = event.dataTransfer?.getData('text/plain')
  if (stepName && draggedStepName.value) {
    // Insert at specific position
    const stepInfo = preprocessingStore.availableSteps.find(s => s.name === stepName)
    if (stepInfo) {
      const defaultArgs: Record<string, any> = {}
      for (const param of stepInfo.parameters) {
        defaultArgs[param.name] = param.default
      }

      const newStep: PipelineStep = {
        id: `step-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: stepName,
        args: defaultArgs,
      }

      preprocessingStore.pipeline.splice(index, 0, newStep)
    }
  }

  draggedStepName.value = null
  dragOverIndex.value = null
}

function onDropAtEnd(event: DragEvent) {
  event.preventDefault()

  const stepName = event.dataTransfer?.getData('text/plain')
  if (stepName) {
    preprocessingStore.addStep(stepName)
  }

  draggedStepName.value = null
  dragOverIndex.value = null
}

// Pipeline reordering
function onPipelineDragStart(event: DragEvent, index: number) {
  event.dataTransfer!.effectAllowed = 'move'
  event.dataTransfer!.setData('pipeline-index', index.toString())
}

function onPipelineDrop(event: DragEvent, toIndex: number) {
  event.preventDefault()

  const fromIndex = parseInt(event.dataTransfer?.getData('pipeline-index') ?? '-1')
  if (fromIndex >= 0 && fromIndex !== toIndex) {
    preprocessingStore.reorderSteps(fromIndex, toIndex)
  }

  dragOverIndex.value = null
}

// Step configuration
function openStepConfig(step: PipelineStep) {
  configStep.value = step
  showStepConfig.value = true
}

function closeStepConfig() {
  showStepConfig.value = false
  configStep.value = null
}

function updateStepArg(argName: string, value: any) {
  if (configStep.value) {
    preprocessingStore.updateStepArgs(configStep.value.id, { [argName]: value })
  }
}

// Preset handling
function loadPreset(presetName: string) {
  preprocessingStore.loadPreset(presetName)
  selectedPreset.value = presetName
}

// Run pipeline
async function runPipeline() {
  if (!sourceStore.currentSource) return

  try {
    // Pass input_path if running on a preprocessed dataset
    const options = inputType.value === 'dataset' && selectedDataset.value
      ? { inputPath: selectedDataset.value.path }
      : {}
    await preprocessingStore.runPipeline(sourceStore.currentSource, options)
    // Start polling for status
    pollJobStatus()
  } catch (error) {
    console.error('Failed to run pipeline:', error)
  }
}

function pollJobStatus() {
  if (!sourceStore.currentSource || !preprocessingStore.runningJob) return

  const interval = setInterval(async () => {
    await preprocessingStore.checkJobStatus(sourceStore.currentSource!)

    if (!preprocessingStore.runningJob ||
        preprocessingStore.runningJob.status === 'completed' ||
        preprocessingStore.runningJob.status === 'failed') {
      clearInterval(interval)
    }
  }, 2000)
}

// Export CLI command
const cliCommand = computed(() => {
  if (!sourceStore.currentSource) return ''
  return preprocessingStore.exportAsCLI(sourceStore.currentSource)
})

function copyCLICommand() {
  navigator.clipboard.writeText(cliCommand.value)
}

// Helper to get step info
function getStepInfo(stepName: string): PreprocessingStepInfo | undefined {
  return preprocessingStore.availableSteps.find(s => s.name === stepName)
}
</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Preprocessing"
            description="Build and preview text preprocessing pipelines"
            icon="preprocessing"
          />
        </div>

        <div class="flex items-center gap-3">
        <!-- Input Selector Button -->
        <button
          @click="showInputSelector = true"
          class="h-9 px-3 rounded-md border border-input bg-background text-sm shadow-sm hover:bg-accent flex items-center gap-2"
        >
          <Database class="h-4 w-4 text-muted-foreground" />
          <span class="max-w-[200px] truncate">{{ currentInputDisplay }}</span>
          <ChevronDown class="h-4 w-4 text-muted-foreground" />
        </button>

        <!-- Preset Selector -->
        <select
          v-model="selectedPreset"
          @change="selectedPreset && loadPreset(selectedPreset)"
          class="h-9 px-3 py-1 text-sm"
        >
          <option :value="null">Load Preset...</option>
          <optgroup label="General">
            <option v-for="preset in preprocessingStore.generalPresets" :key="preset.name" :value="preset.name">
              {{ preset.name }} - {{ preset.description }}
            </option>
          </optgroup>
          <optgroup label="Analysis-Specific">
            <option v-for="preset in preprocessingStore.analysisPresets" :key="preset.name" :value="preset.name">
              {{ preset.name }} - {{ preset.description }}
            </option>
          </optgroup>
        </select>

        <button
          @click="preprocessingStore.clearPipeline()"
          class="h-9 px-3 rounded-md border border-input bg-background text-sm hover:bg-accent"
          :disabled="preprocessingStore.pipelineIsEmpty"
        >
          Clear
        </button>
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 pb-6 space-y-6">
    <!-- Available Steps Palette (collapsible) -->
    <div class="rounded-lg border bg-card">
      <button
        @click="preprocessingStore.stepPaletteCollapsed = !preprocessingStore.stepPaletteCollapsed"
        class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
      >
        <h2 class="text-sm font-medium text-muted-foreground">Available Steps</h2>
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted-foreground">
            {{ preprocessingStore.availableSteps.length }} steps
          </span>
          <component :is="preprocessingStore.stepPaletteCollapsed ? ChevronRight : ChevronDown" class="h-4 w-4 text-muted-foreground" />
        </div>
      </button>

      <div v-if="!preprocessingStore.stepPaletteCollapsed" class="px-4 pb-4">
        <div v-if="preprocessingStore.isLoadingSteps" class="flex items-center justify-center py-8">
          <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
        </div>

        <div v-else class="space-y-4">
          <div
            v-for="[category, steps] in preprocessingStore.stepsByCategory"
            :key="category"
            class="space-y-2"
          >
            <div class="flex items-center gap-2 text-sm font-medium" :style="{ color: CATEGORY_INFO[category]?.getColor() }">
              <component :is="categoryIcons[category]" class="h-4 w-4" />
              {{ CATEGORY_INFO[category]?.label || category }}
            </div>

            <div class="flex flex-wrap gap-2">
              <div
                v-for="step in steps"
                :key="step.name"
                draggable="true"
                @dragstart="onDragStartFromPalette($event, step.name)"
                class="flex items-center gap-2 px-3 py-1.5 rounded-md border bg-background cursor-grab hover:bg-accent transition-colors group relative"
                :class="{ 'opacity-50': step.is_slow }"
                :title="step.description"
              >
                <span class="text-sm">{{ step.display_name }}</span>
                <Clock v-if="step.is_slow" class="h-3 w-3 text-muted-foreground" title="May be slow" />
                <!-- Show parameter count badge if step has options -->
                <span
                  v-if="step.parameters.length > 0"
                  class="text-xs px-1 py-0.5 rounded bg-muted text-muted-foreground"
                  :title="step.parameters.map(p => p.name).join(', ')"
                >
                  {{ step.parameters.length }} opt
                </span>
                <!-- Tooltip showing available options on hover -->
                <div
                  v-if="step.parameters.length > 0"
                  class="absolute left-0 top-full mt-1 z-10 hidden group-hover:block bg-popover border rounded-md shadow-md p-2 min-w-[200px]"
                >
                  <div class="text-xs font-medium mb-1">Options:</div>
                  <div v-for="param in step.parameters" :key="param.name" class="text-xs text-muted-foreground">
                    <span class="font-mono">{{ param.name }}</span>
                    <span v-if="param.type === 'select'" class="ml-1">({{ param.options?.join(' | ') }})</span>
                    <span v-else-if="param.type === 'bool'" class="ml-1">(true/false)</span>
                    <span v-else-if="param.type === 'int' || param.type === 'float'" class="ml-1">
                      ({{ param.type }}{{ param.min_value !== undefined ? `, min: ${param.min_value}` : '' }}{{ param.max_value !== undefined ? `, max: ${param.max_value}` : '' }})
                    </span>
                    <span v-if="param.default !== null && param.default !== undefined" class="ml-1 text-muted-foreground/70">
                      = {{ param.default }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Active Pipeline -->
    <div class="rounded-lg border bg-card">
      <button
        @click="pipelineCollapsed = !pipelineCollapsed"
        class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
      >
        <h2 class="text-sm font-medium text-muted-foreground">Active Pipeline</h2>
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted-foreground">{{ preprocessingStore.pipeline.length }} steps</span>
          <component :is="pipelineCollapsed ? ChevronRight : ChevronDown" class="h-4 w-4 text-muted-foreground" />
        </div>
      </button>

      <div
        v-if="!pipelineCollapsed"
        class="px-4 pb-4"
        @dragover.prevent="dragOverIndex = preprocessingStore.pipeline.length"
        @dragleave="dragOverIndex = null"
        @drop="onDropAtEnd"
      >
        <div v-if="preprocessingStore.pipelineIsEmpty" class="flex items-center justify-center py-8 text-muted-foreground border-2 border-dashed rounded-lg">
          <p>Drag steps here to build your pipeline</p>
        </div>

      <div v-else class="flex flex-wrap items-center gap-2">
        <template v-for="(step, index) in preprocessingStore.pipeline" :key="step.id">
          <!-- Drop zone indicator -->
          <div
            v-if="dragOverIndex === index"
            class="w-1 h-12 bg-primary rounded-full animate-pulse"
          />

          <div
            draggable="true"
            @dragstart="onPipelineDragStart($event, index)"
            @dragover="onDragOver($event, index)"
            @dragleave="onDragLeave"
            @drop="draggedStepName ? onDrop($event, index) : onPipelineDrop($event, index)"
            class="flex items-center gap-1 px-3 py-2 rounded-md border bg-background group"
            :style="{ borderColor: CATEGORY_INFO[getStepInfo(step.name)?.category ?? 'other']?.getColor() }"
          >
            <GripVertical class="h-4 w-4 text-muted-foreground cursor-grab" />
            <span class="text-sm font-medium">{{ getStepInfo(step.name)?.display_name || step.name }}</span>

            <!-- Show active config options as badges -->
            <template v-if="Object.keys(step.args).length > 0">
              <span
                v-for="(value, key) in step.args"
                :key="key"
                class="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground"
                :title="`${key}: ${value}`"
              >
                {{ key }}={{ typeof value === 'boolean' ? (value ? '✓' : '✗') : value }}
              </span>
            </template>

            <button
              v-if="(getStepInfo(step.name)?.parameters.length ?? 0) > 0"
              @click="openStepConfig(step)"
              class="p-1 hover:bg-accent rounded opacity-0 group-hover:opacity-100 transition-opacity"
              title="Configure"
            >
              <Settings class="h-3 w-3" />
            </button>

            <button
              @click="preprocessingStore.removeStep(step.id)"
              class="p-1 hover:bg-destructive hover:text-destructive-foreground rounded opacity-0 group-hover:opacity-100 transition-opacity"
              title="Remove"
            >
              <X class="h-3 w-3" />
            </button>
          </div>

          <ArrowRight v-if="index < preprocessingStore.pipeline.length - 1" class="h-4 w-4 text-muted-foreground" />
        </template>

        <!-- Final drop zone -->
        <div
          v-if="dragOverIndex === preprocessingStore.pipeline.length"
          class="w-1 h-12 bg-primary rounded-full animate-pulse"
        />
      </div>

        <!-- Pipeline info -->
        <div v-if="!preprocessingStore.pipelineIsEmpty" class="flex items-center gap-4 mt-4 pt-4 border-t text-sm text-muted-foreground">
          <span v-if="preprocessingStore.hasSlowSteps" class="flex items-center gap-1 text-warning">
            <Clock class="h-4 w-4" />
            Contains slow steps
          </span>
        </div>
      </div>
    </div>

    <!-- Preview Section -->
    <div class="rounded-lg border bg-card">
      <button
        @click="previewCollapsed = !previewCollapsed"
        class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
      >
        <h2 class="text-sm font-medium text-muted-foreground">Preview</h2>
        <div class="flex items-center gap-2">
          <span v-if="preprocessingStore.batchPreviewResult" class="text-xs text-muted-foreground">
            {{ preprocessingStore.batchPreviewResult.total_remaining }} samples
          </span>
          <component :is="previewCollapsed ? ChevronRight : ChevronDown" class="h-4 w-4 text-muted-foreground" />
        </div>
      </button>

      <div v-if="!previewCollapsed" class="p-4 pt-0">
        <div class="grid grid-cols-2 gap-4">
      <!-- Original Samples -->
      <div class="rounded-lg border bg-card">
        <div class="px-4 border-b h-14 flex items-center">
          <div class="flex items-center justify-between w-full">
            <h2 class="text-sm font-medium">Original Samples</h2>
            <button
              @click="loadSamplesForCurrentPipeline()"
              class="px-2 py-1 text-xs border rounded hover:bg-accent flex items-center gap-1"
              :disabled="!sourceStore.currentSource"
            >
              <RefreshCw class="h-3 w-3" />
              Refresh
            </button>
          </div>
        </div>

        <div
          ref="originalScrollContainer"
          class="p-4 max-h-[400px] overflow-y-auto"
          @scroll="onOriginalScroll"
        >
          <div v-if="preprocessingStore.previewSamples.length > 0" class="space-y-3">
            <div
              v-for="(sample, idx) in preprocessingStore.previewSamples"
              :key="idx"
              class="p-3 rounded border transition-colors"
              :class="sample.filtered ? 'bg-warning/10 border-warning/30' : 'bg-muted/50'"
            >
              <div class="flex items-start gap-3">
                <span class="text-xs font-mono text-muted-foreground shrink-0">{{ idx + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <p class="text-sm" :class="sample.filtered ? 'line-through text-muted-foreground' : ''">{{ sample.text }}</p>
                  <p v-if="sample.date || sample.page_number" class="text-xs text-muted-foreground mt-1">
                    <span v-if="sample.date">{{ sample.date }}</span>
                    <span v-if="sample.date && sample.page_number"> • </span>
                    <span v-if="sample.page_number">Page {{ sample.page_number }}</span>
                  </p>
                </div>
                <span v-if="sample.filtered" class="text-xs px-1.5 py-0.5 rounded bg-warning/20 text-warning shrink-0">filtered</span>
              </div>
            </div>
          </div>

          <div v-else class="text-center text-muted-foreground py-8">
            {{ sourceStore.currentSource ? 'Loading samples...' : 'Select a source to load samples' }}
          </div>
        </div>
      </div>

      <!-- Processed Samples -->
      <div class="rounded-lg border bg-card">
        <div class="px-4 border-b h-14 flex items-center">
          <div class="flex items-center justify-between w-full">
            <h2 class="text-sm font-medium">After Pipeline</h2>
            <div v-if="preprocessingStore.batchPreviewResult" class="flex items-center gap-2 text-xs">
              <span v-if="preprocessingStore.batchPreviewResult.total_removed > 0" class="text-warning flex items-center gap-1">
                <Minus class="h-3 w-3" />
                {{ preprocessingStore.batchPreviewResult.total_removed }} filtered
              </span>
              <span class="text-muted-foreground">
                {{ preprocessingStore.batchPreviewResult.total_remaining }} remaining
              </span>
            </div>
          </div>
        </div>

        <div
          ref="processedScrollContainer"
          class="p-4 max-h-[400px] overflow-y-auto"
          @scroll="onProcessedScroll"
        >
          <div v-if="preprocessingStore.isLoadingPreview" class="flex items-center justify-center py-8">
            <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
          </div>

          <div v-else-if="preprocessingStore.batchPreviewResult?.processed_samples.length" class="space-y-3">
            <div
              v-for="(sample, idx) in preprocessingStore.batchPreviewResult.processed_samples"
              :key="idx"
              class="p-3 rounded border transition-colors"
              :class="sample.filtered ? 'bg-muted/30 border-dashed' : 'bg-muted/50'"
            >
              <div class="flex items-start gap-3">
                <span class="text-xs font-mono text-muted-foreground shrink-0">{{ idx + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <template v-if="sample.filtered">
                    <!-- Skeleton placeholder for filtered row -->
                    <div class="flex items-center gap-2 text-muted-foreground">
                      <Minus class="h-3 w-3 text-warning" />
                      <span class="text-xs italic">Row filtered out</span>
                    </div>
                    <!-- Spacer to match original row height -->
                    <p class="text-xs mt-1">&nbsp;</p>
                  </template>
                  <template v-else>
                    <p class="text-sm">{{ sample.text }}</p>
                    <!-- Spacer to match original row height with metadata -->
                    <p class="text-xs text-muted-foreground mt-1">&nbsp;</p>
                  </template>
                </div>
              </div>
            </div>
          </div>

          <div v-else class="text-center text-muted-foreground py-8">
            {{ preprocessingStore.pipelineIsEmpty ? 'Add steps to see preview' : 'No results' }}
          </div>
        </div>
      </div>
        </div>
      </div>
    </div>

    <!-- Step-by-Step Diff (collapsible) -->
    <div v-if="preprocessingStore.previewResult?.intermediate_steps.length" class="rounded-lg border bg-card">
      <button
        @click="showIntermediateSteps = !showIntermediateSteps"
        class="w-full flex items-center justify-between p-4 text-left"
      >
        <h2 class="text-sm font-medium">Step-by-Step Results</h2>
        <component :is="showIntermediateSteps ? ChevronDown : ChevronRight" class="h-4 w-4 text-muted-foreground" />
      </button>

      <div v-if="showIntermediateSteps" class="px-4 pb-4 space-y-3">
        <div
          v-for="(result, index) in preprocessingStore.previewResult.intermediate_steps"
          :key="index"
          class="p-3 rounded-md border bg-background"
        >
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium">
              Step {{ index + 1 }}: {{ getStepInfo(result.step_name)?.display_name || result.step_name }}
            </span>
            <span class="text-xs text-muted-foreground">{{ result.changes_description }}</span>
          </div>
          <div class="text-sm font-mono text-muted-foreground truncate">
            {{ result.output }}
          </div>
        </div>
      </div>
    </div>

    <!-- Export / Run Section -->
    <div class="rounded-lg border bg-card">
      <button
        @click="runSectionCollapsed = !runSectionCollapsed"
        class="w-full flex items-center justify-between p-4 text-left hover:bg-accent/50 transition-colors"
      >
        <h2 class="text-sm font-medium text-muted-foreground">Run on Full Dataset</h2>
        <component :is="runSectionCollapsed ? ChevronRight : ChevronDown" class="h-4 w-4 text-muted-foreground" />
      </button>

      <div v-if="!runSectionCollapsed" class="px-4 pb-4">
        <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <div class="text-sm">
            <span class="text-muted-foreground">Input:</span>
            <span class="font-medium ml-2">{{ currentInputDisplay }}</span>
            <span v-if="inputType === 'dataset'" class="ml-2 text-xs px-1.5 py-0.5 rounded bg-info/20 text-info">preprocessed</span>
          </div>

          <button
            @click="showCLIExport = !showCLIExport"
            class="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
          >
            <Terminal class="h-4 w-4" />
            CLI Command
          </button>
        </div>

        <button
          @click="runPipeline"
          :disabled="preprocessingStore.pipelineIsEmpty || !sourceStore.currentSource || !!preprocessingStore.runningJob"
          class="flex items-center gap-2 h-9 px-4 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Loader2 v-if="preprocessingStore.runningJob" class="h-4 w-4 animate-spin" />
          <Play v-else class="h-4 w-4" />
          Run Pipeline
        </button>
      </div>

      <!-- CLI Export -->
      <div v-if="showCLIExport" class="mt-4 p-3 rounded-md bg-muted">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs text-muted-foreground">CLI Command</span>
          <button @click="copyCLICommand" class="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1">
            <Copy class="h-3 w-3" />
            Copy
          </button>
        </div>
        <code class="text-xs font-mono break-all">{{ cliCommand }}</code>
      </div>

      <!-- Job Status -->
      <div v-if="preprocessingStore.runningJob" class="mt-4 p-3 rounded-md border">
        <div class="flex items-center justify-between mb-2">
          <span class="text-sm font-medium">
            Job: {{ preprocessingStore.runningJob.job_id }}
          </span>
          <span
            class="text-xs px-2 py-0.5 rounded-full"
            :class="{
              'bg-warning/20 text-warning': preprocessingStore.runningJob.status === 'pending',
              'bg-info/20 text-info': preprocessingStore.runningJob.status === 'running',
              'bg-success/20 text-success': preprocessingStore.runningJob.status === 'completed',
              'bg-destructive/20 text-destructive': preprocessingStore.runningJob.status === 'failed',
            }"
          >
            {{ preprocessingStore.runningJob.status }}
          </span>
        </div>

        <div v-if="preprocessingStore.runningJob.status === 'running'" class="w-full bg-muted rounded-full h-2">
          <div
            class="bg-primary h-2 rounded-full transition-all"
            :style="{ width: `${preprocessingStore.runningJob.progress * 100}%` }"
          />
        </div>

        <div v-if="preprocessingStore.runningJob.error" class="mt-2 text-sm text-destructive">
          {{ preprocessingStore.runningJob.error }}
        </div>

        <div v-if="preprocessingStore.runningJob.output_path" class="mt-2 text-xs text-muted-foreground">
          Output: {{ preprocessingStore.runningJob.output_path }}
        </div>
        </div>
      </div>
    </div>
    </div>

    <!-- Step Configuration Modal -->
    <Teleport to="body">
      <div
        v-if="showStepConfig && configStep"
        class="fixed inset-0 bg-overlay-light flex items-center justify-center z-[100]"
        @click.self="closeStepConfig"
      >
        <div class="bg-card rounded-lg border shadow-lg w-full max-w-md p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold">
            Configure: {{ getStepInfo(configStep.name)?.display_name }}
          </h3>
          <button @click="closeStepConfig" class="p-1 hover:bg-accent rounded">
            <X class="h-5 w-5" />
          </button>
        </div>

        <div class="space-y-4">
          <div
            v-for="param in getStepInfo(configStep.name)?.parameters"
            :key="param.name"
            class="space-y-2"
          >
            <label class="text-sm font-medium">{{ param.name }}</label>
            <p class="text-xs text-muted-foreground">{{ param.description }}</p>

            <!-- Boolean -->
            <input
              v-if="param.type === 'bool'"
              type="checkbox"
              :checked="configStep.args[param.name]"
              @change="updateStepArg(param.name, ($event.target as HTMLInputElement).checked)"
              class="h-4 w-4 rounded border-input"
            />

            <!-- Select -->
            <select
              v-else-if="param.type === 'select'"
              :value="configStep.args[param.name]"
              @change="updateStepArg(param.name, ($event.target as HTMLSelectElement).value)"
              class="w-full h-9 px-3 py-1 text-sm"
            >
              <option v-for="opt in param.options" :key="opt" :value="opt">{{ opt }}</option>
            </select>

            <!-- Number -->
            <input
              v-else-if="param.type === 'int' || param.type === 'float'"
              type="number"
              :value="configStep.args[param.name]"
              :min="param.min_value"
              :max="param.max_value"
              :step="param.type === 'float' ? 0.1 : 1"
              @input="updateStepArg(param.name, param.type === 'int' ? parseInt(($event.target as HTMLInputElement).value) : parseFloat(($event.target as HTMLInputElement).value))"
              class="w-full h-9 rounded-md border border-input bg-background px-3 py-1 text-sm"
            />

            <!-- String -->
            <input
              v-else
              type="text"
              :value="configStep.args[param.name]"
              @input="updateStepArg(param.name, ($event.target as HTMLInputElement).value)"
              class="w-full h-9 rounded-md border border-input bg-background px-3 py-1 text-sm"
            />
          </div>
        </div>

        <div class="flex justify-end mt-6">
          <button
            @click="closeStepConfig"
            class="h-9 px-4 rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
          >
            Done
          </button>
        </div>
        </div>
      </div>
    </Teleport>

    <!-- Input Selector Dialog -->
    <Teleport to="body">
      <div
        v-if="showInputSelector"
        class="fixed inset-0 bg-overlay-light flex items-center justify-center z-[100]"
        @click.self="showInputSelector = false"
      >
        <div class="bg-card rounded-lg border shadow-lg w-full max-w-lg p-6">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-lg font-semibold">Select Input</h3>
          <button @click="showInputSelector = false" class="p-1 hover:bg-accent rounded">
            <X class="h-5 w-5" />
          </button>
        </div>

        <!-- Raw Sources Section -->
        <div class="mb-6">
          <h4 class="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
            <Database class="h-4 w-4" />
            Raw Sources
          </h4>
          <div class="grid grid-cols-2 gap-2">
            <button
              v-for="source in sourceStore.sources"
              :key="source"
              @click="selectSourceInput(source)"
              class="p-3 rounded-md border bg-background hover:bg-accent text-left transition-colors"
              :class="{ 'ring-2 ring-primary': inputType === 'source' && sourceStore.currentSource === source }"
            >
              <span class="text-sm font-medium">{{ source }}</span>
            </button>
          </div>
        </div>

        <!-- Preprocessed Datasets Section -->
        <div>
          <h4 class="text-sm font-medium text-muted-foreground mb-3 flex items-center gap-2">
            <FileText class="h-4 w-4" />
            Preprocessed Datasets
          </h4>

          <div v-if="preprocessedDatasets.length > 0" class="space-y-2 max-h-[250px] overflow-y-auto">
            <button
              v-for="dataset in preprocessedDatasets"
              :key="dataset.path"
              @click="selectDatasetInput(dataset)"
              class="w-full p-3 rounded-md border bg-background hover:bg-accent text-left transition-colors"
              :class="{ 'ring-2 ring-primary': inputType === 'dataset' && selectedDataset?.path === dataset.path }"
            >
              <div class="flex items-center justify-between">
                <span class="text-sm font-medium">{{ dataset.name }}</span>
                <span class="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">{{ dataset.source }}</span>
              </div>
              <p class="text-xs text-muted-foreground mt-1">{{ dataset.steps }} steps • {{ dataset.created }}</p>
            </button>
          </div>

          <div v-else class="text-center text-muted-foreground py-6 border-2 border-dashed rounded-lg">
            <p class="text-sm">No preprocessed datasets yet</p>
            <p class="text-xs mt-1">Run a pipeline to create one</p>
          </div>
        </div>

        <div class="flex justify-end mt-6">
          <button
            @click="showInputSelector = false"
            class="h-9 px-4 rounded-md border border-input bg-background hover:bg-accent"
          >
            Cancel
          </button>
        </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>
