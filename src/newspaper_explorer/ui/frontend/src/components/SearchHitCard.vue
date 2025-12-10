<script setup lang="ts">
import { computed } from 'vue'
import { formatDate, parsePageMetadata, type PageMetadata } from '@/lib/composables/useImageUtils'
import { Button } from '@/components/ui/button'

export interface SearchResult {
  text_block_id: string
  page_id: string
  date: string
  text: string
  highlights: string[]
  score: number
  x?: number
  y?: number
  width?: number
  height?: number
  image_path?: string
}

interface Props {
  result: SearchResult
}

const props = defineProps<Props>()

const emit = defineEmits<{
  'view-page': [result: SearchResult]
}>()

const metadata = computed<PageMetadata | null>(() => {
  return parsePageMetadata(props.result.page_id)
})

const formattedDate = computed(() => {
  return formatDate(props.result.date)
})

const metadataDisplay = computed(() => {
  if (!metadata.value) return props.result.page_id
  return `Issue ${metadata.value.issue} - Daily ${metadata.value.daily} - Page ${metadata.value.page}`
})

// Highlight text logic
const highlightedText = computed(() => {
  // If we have highlights, show them joined by ...
  // Otherwise show truncated text
  if (props.result.highlights && props.result.highlights.length > 0) {
    return props.result.highlights.map(h => `...${h}...`).join('<br/>')
  }
  return props.result.text.length > 300 ? props.result.text.substring(0, 300) + '...' : props.result.text
})
</script>

<template>
  <div class="rounded-lg border bg-card p-4 hover:shadow-md transition-shadow">
    <div class="flex justify-between items-start mb-2">
      <div class="space-y-1">
        <div class="font-medium text-lg">{{ formattedDate }}</div>
        <div class="text-sm text-muted-foreground">{{ metadataDisplay }}</div>
      </div>
      <Button
        @click="emit('view-page', result)"
        variant="outline"
        size="sm"
      >
        View Page
      </Button>
    </div>

    <div class="mt-3 text-sm leading-relaxed font-serif bg-muted/30 p-3 rounded border border-border/50">
      <div v-html="highlightedText"></div>
    </div>

    <div class="mt-2 flex items-center gap-2">
      <span class="text-xs text-muted-foreground font-mono">{{ result.text_block_id }}</span>
    </div>
  </div>
</template>
