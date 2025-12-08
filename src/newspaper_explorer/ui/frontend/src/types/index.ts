export interface AnalysisResultSummary {
  count: number
  parquet: number
  csv: number
}

export interface SourceInfo {
  name: string
  dataset_name: string
  data_type: string
  metadata: Record<string, any>
  loading: Record<string, any>
  has_text: boolean
  has_entities: boolean
  has_keywords: boolean
  has_layout: boolean
  has_topics: boolean
  has_emotions: boolean
  has_concepts: boolean
  has_images: boolean
  total_archive_size?: string  // Compressed XML archive size
  image_size?: string          // Total size of downloaded images
  image_count?: number         // Number of downloaded images
  analysis_results: Record<string, AnalysisResultSummary>
}

export interface SourceStats {
  total_issues: number
  total_pages: number
  total_lines: number
  total_blocks: number
  total_images: number
  date_range: [string, string]
  years_available: number[]
}

export interface Issue {
  issue_id: string
  date: string
  newspaper_title: string
  year_volume: string
  page_count: number
  has_images: boolean
}

export interface Page {
  page_id: string
  issue_id: string
  date: string
  newspaper_title: string
  page_number: number
  text_preview?: string
  image_url?: string
  has_image: boolean
}

export interface TextBlock {
  text_block_id: string
  page_id: string
  issue_id: string
  date: string
  text: string
  x: number
  y: number
  width: number
  height: number
}

export interface Entity {
  entity_text: string
  entity_type: string
  detection_count: number
  avg_confidence: number
  min_confidence: number
  max_confidence: number
  line_ids: string[]
}

export interface EntityOccurrence {
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

export interface Keyword {
  keyword: string
  frequency: number
  tfidf_score: number
}

export interface LayoutRegion {
  region_id: string
  page_id: string
  label: string
  confidence: number
  x: number
  y: number
  width: number
  height: number
  text?: string
  image_url?: string
}

export interface Topic {
  topic_id: number
  label: string
  top_words: string[]
  document_count: number
}

export interface EmotionScore {
  text_block_id: string
  date: string
  emotion: string
  score: number
  text_preview: string
}

export interface Concept {
  concept: string
  frequency: number
  category?: string
}

export interface ConceptRelation {
  source: string
  target: string
  weight: number
  relation_type?: string
}

export interface SearchResult {
  text_block_id: string
  page_id: string
  date: string
  text: string
  highlights: string[]
  score: number
}

export interface SearchResponse {
  total: number
  results: SearchResult[]
  page: number
  page_size: number
}
