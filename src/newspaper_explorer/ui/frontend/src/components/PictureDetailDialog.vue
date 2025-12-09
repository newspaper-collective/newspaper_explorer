<script setup lang="ts">
import { ref, computed } from 'vue'
import { X, ChevronLeft, ChevronRight, Image } from 'lucide-vue-next'
import OpenSeadragonViewer from './OpenSeadragonViewer.vue'

export interface Picture {
  detection_id: string
  page_id: string
  class_name: string
  confidence: number
  bbox_x1: number
  bbox_y1: number
  bbox_x2: number
  bbox_y2: number
  date?: string
  image_path?: string
  newspaper_title?: string
  text_content?: string
  caption_bbox?: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

export interface PageMetadata {
  date: string
  issue_number: string
  daily_count: string
  page_number: string
}

export interface Detection {
  detection_id: string
  class_name: string
  confidence: number
  bbox: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

interface Props {
  open: boolean
  picture: Picture | null
  source: string
  imagePath: string
  pageMetadata: PageMetadata | null
  allPageDetections: Detection[]
  croppedImages: Record<string, string>
  croppedCaptions: Record<string, string>
  cropLoadingImages: Set<string>
  cropLoadingCaptions: Set<string>
  pageThumbnail: string
  pageThumbnailLoading: boolean
  hasPrevious: boolean
  hasNext: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  close: []
  previous: []
  next: []
}>()

const showAllDetections = ref(false)

const pictureConfidence = computed(() => {
  if (!props.picture) return ''
  return (props.picture.confidence * 100).toFixed(1) + '%'
})

const pictureSize = computed(() => {
  if (!props.picture) return ''
  const width = (props.picture.bbox_x2 - props.picture.bbox_x1).toFixed(0)
  const height = (props.picture.bbox_y2 - props.picture.bbox_y1).toFixed(0)
  return `${width}×${height}px`
})

const croppedImage = computed(() => {
  if (!props.picture) return undefined
  return props.croppedImages[props.picture.detection_id]
})

const croppedCaption = computed(() => {
  if (!props.picture) return undefined
  return props.croppedCaptions[`caption_${props.picture.detection_id}`]
})

const isLoadingImage = computed(() => {
  if (!props.picture) return false
  return props.cropLoadingImages.has(props.picture.detection_id)
})

const isLoadingCaption = computed(() => {
  if (!props.picture) return false
  return props.cropLoadingCaptions.has(`caption_${props.picture.detection_id}`)
})

const captionText = computed(() => {
  return props.picture?.text_content || null
})

const pictureCount = computed(() => {
  return props.allPageDetections.filter(d => d.class_name === 'Picture').length
})

const captionCount = computed(() => {
  return props.allPageDetections.filter(d => d.class_name === 'Caption').length
})
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-[100] flex items-center justify-center bg-overlay-medium backdrop-blur-sm"
      @click="emit('close')"
    >
      <div
        class="relative w-full max-w-7xl h-[90vh] flex flex-col bg-background rounded-lg shadow-lg overflow-hidden"
        @click.stop
      >
      <!-- Header -->
      <div class="flex items-center justify-between px-4 py-2 border-b bg-muted/50">
        <div v-if="pageMetadata" class="flex items-center gap-3">
          <span class="font-semibold text-sm">
            {{ pageMetadata.date }}
          </span>
          <span class="text-xs text-muted-foreground border-l pl-3">
            Issue {{ pageMetadata.issue_number }} • Daily {{ pageMetadata.daily_count }} • Page {{
              pageMetadata.page_number
            }}
          </span>
        </div>
        <h2 v-else class="text-sm font-semibold">Picture Details</h2>
        <div class="flex items-center gap-1">
          <button
            @click="emit('previous')"
            :disabled="!hasPrevious"
            class="p-1.5 hover:bg-accent rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Previous picture (Left Arrow)"
          >
            <ChevronLeft class="w-4 h-4" />
          </button>
          <button
            @click="emit('next')"
            :disabled="!hasNext"
            class="p-1.5 hover:bg-accent rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Next picture (Right Arrow)"
          >
            <ChevronRight class="w-4 h-4" />
          </button>
          <div class="w-px h-4 bg-border mx-2"></div>
          <button
            @click="emit('close')"
            class="p-1.5 hover:bg-accent rounded-md transition-colors"
          >
            <X class="w-4 h-4" />
          </button>
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-6">
        <div class="flex gap-6">
          <!-- Left sidebar -->
          <div class="flex-shrink-0 space-y-4" style="width: 300px">
            <!-- Overview section -->
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                Overview
              </h3>
              <label class="flex items-center gap-2 text-xs cursor-pointer">
                <input v-model="showAllDetections" type="checkbox" class="rounded border-input" />
                <span>Show all</span>
              </label>
            </div>

            <!-- Page thumbnail -->
            <div
              class="rounded-lg border bg-black overflow-hidden relative"
              style="width: 300px; height: 400px"
            >
              <OpenSeadragonViewer
                v-if="imagePath && showAllDetections && allPageDetections.length > 0"
                :image-url="`/static/${source}/images/${imagePath}`"
                :current-page="1"
                :total-pages="1"
                :detections="allPageDetections"
                :show-zoom-controls="false"
                navigator-width="80px"
                navigator-height="60px"
                class="w-full h-full"
              />
              <div
                v-else-if="pageThumbnail"
                class="w-full h-full flex items-center justify-center bg-black"
              >
                <img
                  :src="pageThumbnail"
                  alt="Page thumbnail"
                  class="max-w-full max-h-full object-contain"
                />
              </div>
              <div
                v-else-if="pageThumbnailLoading"
                class="w-full h-full flex items-center justify-center"
              >
                <div
                  class="w-8 h-8 border-2 border-muted-foreground border-t-primary rounded-full animate-spin"
                />
              </div>
              <div
                v-else
                class="w-full h-full flex items-center justify-center text-sm text-muted-foreground"
              >
                No preview
              </div>
            </div>

            <!-- Legend -->
            <div class="text-xs text-muted-foreground space-y-2 border-b pb-4">
              <div class="flex items-center gap-4">
                <div class="flex items-center gap-2">
                  <div class="w-3 h-3 border border-info rounded bg-info/20"></div>
                  <span>Picture</span>
                </div>
                <div class="flex items-center gap-2">
                  <div class="w-3 h-3 border border-success rounded bg-success/20"></div>
                  <span>Caption</span>
                </div>
              </div>
              <div v-if="showAllDetections && allPageDetections.length > 0">
                {{ pictureCount }} pictures, {{ captionCount }} captions
              </div>
            </div>

            <!-- Page metadata -->
            <div v-if="pageMetadata" class="space-y-3 text-sm">
              <div>
                <div class="text-xs text-muted-foreground mb-0.5">Date</div>
                <div class="font-medium">{{ pageMetadata.date }}</div>
              </div>
              <div class="grid grid-cols-3 gap-2">
                <div>
                  <div class="text-xs text-muted-foreground mb-0.5">Issue</div>
                  <div class="font-medium">{{ pageMetadata.issue_number }}</div>
                </div>
                <div>
                  <div class="text-xs text-muted-foreground mb-0.5">Daily</div>
                  <div class="font-medium">{{ pageMetadata.daily_count }}</div>
                </div>
                <div>
                  <div class="text-xs text-muted-foreground mb-0.5">Page</div>
                  <div class="font-medium">{{ pageMetadata.page_number }}</div>
                </div>
              </div>
              <div v-if="picture?.newspaper_title">
                <div class="text-xs text-muted-foreground mb-0.5">Newspaper</div>
                <div class="font-medium">{{ picture.newspaper_title }}</div>
              </div>
            </div>

            <!-- Picture metadata -->
            <div v-if="picture" class="space-y-3 text-sm border-t pt-4">
              <h3 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                Detection Info
              </h3>
              <div class="grid grid-cols-2 gap-2">
                <div>
                  <div class="text-xs text-muted-foreground mb-0.5">Confidence</div>
                  <div class="font-medium">{{ pictureConfidence }}</div>
                </div>
                <div>
                  <div class="text-xs text-muted-foreground mb-0.5">Size</div>
                  <div class="font-medium">{{ pictureSize }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- Right content area -->
          <div class="flex-1 space-y-6">
            <!-- Cropped picture -->
            <div class="space-y-2">
              <h3 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                Extracted Picture
              </h3>
              <div class="rounded-lg border bg-muted/30 overflow-hidden" style="height: 600px">
                <OpenSeadragonViewer
                  v-if="picture && croppedImage"
                  :image-url="croppedImage"
                  :current-page="1"
                  :total-pages="1"
                  :detections="[]"
                />
                <div
                  v-else-if="picture && isLoadingImage"
                  class="w-full h-full flex items-center justify-center"
                >
                  <div
                    class="w-16 h-16 border-4 border-muted-foreground border-t-primary rounded-full animate-spin"
                  />
                </div>
                <div v-else class="w-full h-full flex items-center justify-center">
                  <Image class="w-16 h-16 text-muted-foreground" />
                </div>
              </div>
            </div>

            <!-- Caption section -->
            <div v-if="picture && picture.caption_bbox" class="space-y-2">
              <h3 class="text-sm font-semibold text-muted-foreground uppercase tracking-wide">
                Associated Caption
              </h3>
              <div class="flex items-center justify-center rounded-lg border bg-muted/30 p-4">
                <img
                  v-if="croppedCaption"
                  :src="croppedCaption"
                  alt="Caption"
                  class="max-w-full max-h-64 object-contain rounded"
                />
                <div
                  v-else-if="isLoadingCaption"
                  class="w-12 h-12 border-4 border-muted-foreground border-t-primary rounded-full animate-spin"
                />
                <div v-else class="text-sm text-muted-foreground">Caption image not loaded</div>
              </div>
              <!-- Caption text -->
              <div
                v-if="captionText"
                class="rounded-lg border bg-muted/20 p-3 text-sm leading-relaxed"
              >
                <div class="text-xs font-medium text-muted-foreground mb-1">Extracted Text:</div>
                <div class="text-foreground">{{ captionText }}</div>
              </div>
            </div>

            <!-- No caption message -->
            <div
              v-else-if="picture"
              class="rounded-lg border border-dashed bg-muted/20 p-6 text-center text-sm text-muted-foreground"
            >
              No caption found for this picture
            </div>
          </div>
        </div>
      </div>
      </div>
    </div>
  </Teleport>
</template>
