<script setup lang="ts">
import { computed } from 'vue'
import { formatDate, parsePageMetadata, type PageMetadata } from '@/lib/composables/useImageUtils'

export interface PictureData {
  detection_id: string
  page_id: string
  class_name: string
  date?: string
  confidence: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  image_path?: string
  newspaper_title?: string
  text_content?: string
  caption_id?: string
  caption_bbox?: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

interface Props {
  picture: PictureData
  croppedImage?: string
  croppedCaption?: string
  isLoadingImage?: boolean
  isLoadingCaption?: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  click: [picture: PictureData]
}>()

const metadata = computed<PageMetadata | null>(() => {
  return parsePageMetadata(props.picture.page_id)
})

const formattedDate = computed(() => {
  return formatDate(props.picture.date)
})

const metadataDisplay = computed(() => {
  if (!metadata.value) return props.picture.page_id
  return `Issue ${metadata.value.issue} • Daily ${metadata.value.daily} • Page ${metadata.value.page}` 
})

const pictureSize = computed(() => {
  const width = (props.picture.bbox_x2 - props.picture.bbox_x1).toFixed(0)
  const height = (props.picture.bbox_y2 - props.picture.bbox_y1).toFixed(0)
  return `${width} × ${height}px`
})

const confidencePercent = computed(() => {
  return (props.picture.confidence * 100).toFixed(1) + '%'
})
</script>

<template>
  <div
    class="rounded-lg border bg-background overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
    @click="emit('click', picture)"
  >
    <!-- Cropped picture image -->
    <div class="aspect-[4/3] bg-black flex items-center justify-center">
      <img
        v-if="croppedImage"
        :src="croppedImage"
        :alt="`Picture from ${formattedDate}`"
        class="w-full h-full object-contain"
      />
      <div
        v-else-if="isLoadingImage"
        class="w-16 h-16 border-4 border-muted-foreground border-t-primary rounded-full animate-spin"
      />
      <svg
        v-else
        class="w-16 h-16 text-muted-foreground"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
        />
      </svg>
    </div>

    <!-- Metadata -->
    <div class="p-4 space-y-2">
      <div class="text-sm font-medium">
        {{ formattedDate }}
      </div>
      <div v-if="metadata" class="text-xs text-muted-foreground">
        {{ metadataDisplay }}
      </div>
      <div class="text-xs text-muted-foreground">
        Confidence: {{ confidencePercent }}
      </div>
      <div class="text-xs text-muted-foreground">
        Size: {{ pictureSize }}
      </div>

      <!-- Caption image if available -->
      <div
        v-if="picture.caption_bbox && croppedCaption"
        class="pt-2 mt-2 border-t border-border"
      >
        <div class="text-xs font-medium text-foreground mb-1">Caption:</div>
        <div class="mt-1 rounded overflow-hidden bg-muted flex items-center justify-center">
          <img
            :src="croppedCaption"
            :alt="`Caption for picture from ${formattedDate}`"
            class="max-w-full h-auto"
            style="max-height: 150px"
          />
        </div>
      </div>

      <!-- Loading state for caption -->
      <div
        v-else-if="picture.caption_bbox && isLoadingCaption"
        class="pt-2 mt-2 border-t border-border"
      >
        <div class="text-xs font-medium text-foreground mb-1">Caption:</div>
        <div class="flex items-center justify-center py-4">
          <div
            class="w-8 h-8 border-2 border-muted-foreground border-t-primary rounded-full animate-spin"
          />
        </div>
      </div>
    </div>
  </div>
</template>
