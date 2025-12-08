/**
 * Pinia store for preprocessing pipeline state management
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import api from '@/lib/api'
import type {
  PreprocessingStepInfo,
  PresetInfo,
  PipelineStep,
  PipelineStepConfig,
  PreprocessingPreviewResponse,
  PreprocessingStatusResponse,
  TextSample,
  BatchPreviewResponse,
} from '@/types/preprocessing'

export const usePreprocessingStore = defineStore('preprocessing', () => {
  // State
  const availableSteps = ref<PreprocessingStepInfo[]>([])
  const presets = ref<PresetInfo[]>([])
  const pipeline = ref<PipelineStep[]>([])
  const previewSamples = ref<TextSample[]>([])
  const previewText = ref('')
  const previewResult = ref<PreprocessingPreviewResponse | null>(null)
  const batchPreviewResult = ref<BatchPreviewResponse | null>(null)
  const isLoadingSteps = ref(false)
  const isLoadingPreview = ref(false)
  const runningJob = ref<PreprocessingStatusResponse | null>(null)
  const customTextMode = ref(false)
  const stepPaletteCollapsed = ref(false)

  // Computed
  const stepsByCategory = computed(() => {
    const grouped: Record<string, PreprocessingStepInfo[]> = {}
    for (const step of availableSteps.value) {
      if (!grouped[step.category]) {
        grouped[step.category] = []
      }
      grouped[step.category].push(step)
    }
    // Return as array of [category, steps] tuples in defined order
    const order = ['normalization', 'cleaning', 'filtering', 'modernization', 'linguistic', 'quality']
    const ordered: [string, PreprocessingStepInfo[]][] = []
    for (const cat of order) {
      if (grouped[cat]) {
        ordered.push([cat, grouped[cat]])
      }
    }
    // Add any remaining categories not in order
    for (const cat of Object.keys(grouped)) {
      if (!order.includes(cat)) {
        ordered.push([cat, grouped[cat]])
      }
    }
    return ordered
  })

  const generalPresets = computed(() =>
    presets.value.filter(p => p.category === 'general')
  )

  const analysisPresets = computed(() =>
    presets.value.filter(p => p.category === 'analysis')
  )

  const pipelineIsEmpty = computed(() => pipeline.value.length === 0)

  const hasSlowSteps = computed(() =>
    pipeline.value.some(step => {
      const info = availableSteps.value.find(s => s.name === step.name)
      return info?.is_slow ?? false
    })
  )

  // Actions
  async function loadSteps() {
    if (availableSteps.value.length > 0) return // Already loaded

    isLoadingSteps.value = true
    try {
      const response = await api.get('/preprocessing/steps')
      availableSteps.value = response.data
    } catch (error) {
      console.error('Failed to load preprocessing steps:', error)
    } finally {
      isLoadingSteps.value = false
    }
  }

  async function loadPresets() {
    if (presets.value.length > 0) return // Already loaded

    try {
      const response = await api.get('/preprocessing/presets')
      presets.value = response.data
    } catch (error) {
      console.error('Failed to load preprocessing presets:', error)
    }
  }

  function generateStepId(): string {
    return `step-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  function addStep(stepName: string, args: Record<string, any> = {}) {
    const stepInfo = availableSteps.value.find(s => s.name === stepName)
    if (!stepInfo) return

    // Initialize args with defaults
    const defaultArgs: Record<string, any> = {}
    for (const param of stepInfo.parameters) {
      defaultArgs[param.name] = param.default
    }

    pipeline.value.push({
      id: generateStepId(),
      name: stepName,
      args: { ...defaultArgs, ...args },
    })
  }

  function removeStep(stepId: string) {
    const index = pipeline.value.findIndex(s => s.id === stepId)
    if (index !== -1) {
      pipeline.value.splice(index, 1)
    }
  }

  function reorderSteps(fromIndex: number, toIndex: number) {
    const [removed] = pipeline.value.splice(fromIndex, 1)
    pipeline.value.splice(toIndex, 0, removed)
  }

  function updateStepArgs(stepId: string, args: Record<string, any>) {
    const step = pipeline.value.find(s => s.id === stepId)
    if (step) {
      step.args = { ...step.args, ...args }
    }
  }

  function clearPipeline() {
    pipeline.value = []
    previewResult.value = null
  }

  function loadPreset(presetName: string) {
    const preset = presets.value.find(p => p.name === presetName)
    if (!preset) return

    pipeline.value = preset.steps.map(step => ({
      id: generateStepId(),
      name: step.name,
      args: { ...step.args },
    }))
  }

  async function loadSampleText(sourceName: string, sampleType: 'random' | 'hyphenated' = 'random') {
    try {
      const response = await api.get(`/preprocessing/${sourceName}/sample`, {
        params: { count: 10, sample_type: sampleType },
      })
      if (response.data.length > 0) {
        previewSamples.value = response.data
        // Set first sample as preview text for backwards compatibility
        previewText.value = response.data[0].text
      }
    } catch (error) {
      console.error('Failed to load sample text:', error)
    }
  }

  async function preview(showIntermediate: boolean = true) {
    if (!previewText.value || pipeline.value.length === 0) {
      previewResult.value = null
      return
    }

    isLoadingPreview.value = true
    try {
      const response = await api.post('/preprocessing/preview', {
        text: previewText.value,
        steps: pipeline.value.map(s => ({ name: s.name, args: s.args })),
        show_intermediate: showIntermediate,
      })
      previewResult.value = response.data
    } catch (error) {
      console.error('Failed to preview preprocessing:', error)
      previewResult.value = null
    } finally {
      isLoadingPreview.value = false
    }
  }

  async function previewBatch(showIntermediate: boolean = true) {
    if (previewSamples.value.length === 0 || pipeline.value.length === 0) {
      batchPreviewResult.value = null
      return
    }

    isLoadingPreview.value = true
    try {
      const response = await api.post('/preprocessing/preview-batch', {
        texts: previewSamples.value,
        steps: pipeline.value.map(s => ({ name: s.name, args: s.args })),
        show_intermediate: showIntermediate,
      })
      batchPreviewResult.value = response.data
    } catch (error) {
      console.error('Failed to batch preview preprocessing:', error)
      batchPreviewResult.value = null
    } finally {
      isLoadingPreview.value = false
    }
  }

  async function runPipeline(
    sourceName: string,
    options: {
      textColumn?: string
      outputColumn?: string
      includeOriginal?: boolean
      inputPath?: string
    } = {}
  ) {
    try {
      const response = await api.post(`/preprocessing/${sourceName}/run`, {
        steps: pipeline.value.map(s => ({ name: s.name, args: s.args })),
        text_column: options.textColumn ?? 'text',
        output_column: options.outputColumn ?? 'text_processed',
        include_original: options.includeOriginal ?? true,
        input_path: options.inputPath ?? null,
      })

      runningJob.value = {
        job_id: response.data.job_id,
        status: 'pending',
        progress: 0,
      }

      return response.data
    } catch (error) {
      console.error('Failed to start preprocessing:', error)
      throw error
    }
  }

  async function checkJobStatus(sourceName: string) {
    if (!runningJob.value) return

    try {
      const response = await api.get(
        `/preprocessing/${sourceName}/status/${runningJob.value.job_id}`
      )
      runningJob.value = response.data

      // Clear job if completed or failed
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        setTimeout(() => {
          runningJob.value = null
        }, 5000)
      }
    } catch (error) {
      console.error('Failed to check job status:', error)
    }
  }

  function setPreviewText(text: string) {
    previewText.value = text
    customTextMode.value = true
  }

  function getStepInfo(stepName: string): PreprocessingStepInfo | undefined {
    return availableSteps.value.find(s => s.name === stepName)
  }

  // Export pipeline as CLI command
  function exportAsCLI(sourceName: string): string {
    const stepArgs = pipeline.value.map(s => {
      if (Object.keys(s.args).length === 0) {
        return s.name
      }
      return `${s.name}:${JSON.stringify(s.args)}`
    }).join(',')

    return `newspaper-explorer data preprocess --source ${sourceName} --steps "${stepArgs}"`
  }

  return {
    // State
    availableSteps,
    presets,
    pipeline,
    previewText,
    previewSamples,
    previewResult,
    batchPreviewResult,
    isLoadingSteps,
    isLoadingPreview,
    runningJob,
    customTextMode,
    stepPaletteCollapsed,
    // Computed
    stepsByCategory,
    generalPresets,
    analysisPresets,
    pipelineIsEmpty,
    hasSlowSteps,
    // Actions
    loadSteps,
    loadPresets,
    addStep,
    removeStep,
    reorderSteps,
    updateStepArgs,
    clearPipeline,
    loadPreset,
    loadSampleText,
    preview,
    previewBatch,
    runPipeline,
    checkJobStatus,
    setPreviewText,
    getStepInfo,
    exportAsCLI,
  }
})
