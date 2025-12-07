/**
 * TypeScript types for preprocessing pipeline UI
 */

export interface StepParameter {
  name: string
  type: 'int' | 'float' | 'bool' | 'string' | 'select'
  default: any
  description: string
  options?: string[]
  min_value?: number
  max_value?: number
}

export interface PreprocessingStepInfo {
  name: string
  display_name: string
  description: string
  category: 'normalization' | 'modernization' | 'cleaning' | 'filtering' | 'linguistic' | 'quality'
  is_filter: boolean
  is_slow: boolean
  parameters: StepParameter[]
}

export interface PresetInfo {
  name: string
  description: string
  use_case: string
  steps: PipelineStepConfig[]
  category: 'general' | 'analysis'
}

export interface PipelineStepConfig {
  name: string
  args: Record<string, any>
}

export interface PipelineStep extends PipelineStepConfig {
  id: string // Unique ID for drag-and-drop
}

export interface StepResult {
  step_name: string
  output: string
  changes_description: string
}

export interface PreprocessingPreviewResponse {
  original: string
  final: string
  intermediate_steps: StepResult[]
  stats: {
    char_diff: number
    original_length: number
    final_length: number
    word_diff?: number
  }
}

// Batch preview types for filter statistics
export interface TextSample {
  text: string
  date?: string
  page_number?: number
  filtered?: boolean
}

export interface BatchStepResult {
  step_name: string
  input_count: number
  output_count: number
  removed_count: number
  is_filter: boolean
}

export interface BatchPreviewResponse {
  original_samples: TextSample[]
  processed_samples: TextSample[]
  step_stats: BatchStepResult[]
  total_removed: number
  total_remaining: number
}

export interface PreprocessingRunResponse {
  job_id: string
  estimated_time_seconds: number
  output_path: string
  message: string
}

export interface PreprocessingStatusResponse {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  current_step?: string
  error?: string
  output_path?: string
}

// Category metadata for UI display
export const CATEGORY_ORDER = [
  'normalization',
  'cleaning',
  'filtering',
  'modernization',
  'linguistic',
  'quality',
] as const

export const CATEGORY_INFO: Record<string, { label: string; color: string; icon: string }> = {
  normalization: { label: 'Normalization', color: '#2E5EFF', icon: 'Type' },
  cleaning: { label: 'Cleaning', color: '#00E676', icon: 'Eraser' },
  filtering: { label: 'Filtering', color: '#FF9100', icon: 'Filter' },
  modernization: { label: 'Modernization', color: '#9C27FF', icon: 'Sparkles' },
  linguistic: { label: 'Lemmatization', color: '#FF3333', icon: 'Languages' },
  quality: { label: 'Quality', color: '#00E5FF', icon: 'CheckCircle' },
}

export interface PreprocessedDatasetInfo {
  name: string
  path: string
  created: string
  steps: number
  row_count?: number | null
}
