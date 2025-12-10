<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import OpenSeadragon from 'openseadragon'
import { ZoomIn, ZoomOut, Home, Maximize2, ChevronLeft, ChevronRight } from 'lucide-vue-next'
import { getDetectionColor } from '@/lib/imageAnnotation'
import {
  getHighlightColor,
  getHighlightBorderColor,
  getSelectionLightColor,
  getSelectionMediumColor,
  getSelectionBorderColor,
} from '@/lib/colors'

interface Detection {
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

interface TextLine {
  line_id?: string
  text_block_id?: string
  text: string
  x: number
  y: number
  width: number
  height: number
}

interface Props {
  imageUrl: string
  currentPage: number
  totalPages: number
  detections?: Detection[]
  textLines?: TextLine[]
  allTextLines?: TextLine[]  // Full list for finding highlighted line when overlays are off
  highlightedLineId?: string | null
  imageWidth?: number
  imageHeight?: number
  showZoomControls?: boolean
  showNavigator?: boolean
  navigatorPosition?: 'TOP_LEFT' | 'TOP_RIGHT' | 'BOTTOM_LEFT' | 'BOTTOM_RIGHT' | 'ABSOLUTE'
  navigatorHeight?: string
  navigatorWidth?: string
}

const props = withDefaults(defineProps<Props>(), {
  detections: () => [],
  textLines: () => [],
  allTextLines: () => [],
  highlightedLineId: null,
  showZoomControls: true,
  showNavigator: true,
  navigatorPosition: 'BOTTOM_RIGHT',
  navigatorHeight: '120px',
  navigatorWidth: '150px',
})

const emit = defineEmits<{
  changePage: [delta: number]
  lineClick: [lineId: string]
  lineHover: [lineId: string | null]
}>()

const viewerContainer = ref<HTMLElement | null>(null)
let viewer: OpenSeadragon.Viewer | null = null
const textLineOverlays = new Map<string, HTMLElement>()
const detectionOverlays = new Map<string, HTMLElement>()
let highlightedLineOverlay: HTMLElement | null = null  // Single overlay for highlighted line when overlays are off
let spotlightOverlay: HTMLElement | null = null  // Dark overlay with cutout for spotlight effect
let hoverTimeout: ReturnType<typeof setTimeout> | null = null  // Delay for hover transitions

function addTextLineOverlays() {
  if (!viewer) return

  // Clear existing text line overlays
  textLineOverlays.forEach((element) => {
    viewer?.removeOverlay(element)
  })
  textLineOverlays.clear()

  // If no text lines to show, just return after clearing
  if (!props.textLines || props.textLines.length === 0) {
    return
  }

  const tiledImage = viewer.world.getItemAt(0)
  if (!tiledImage) {
    return
  }

  const imageSize = tiledImage.getContentSize()
  const osdImageWidth = imageSize.x
  const osdImageHeight = imageSize.y

  if (!osdImageWidth || !osdImageHeight) {
    return
  }

  // Use provided ALTO image dimensions if available, otherwise fall back to OSD dimensions
  const altoImageWidth = props.imageWidth || osdImageWidth
  const altoImageHeight = props.imageHeight || osdImageHeight

  // Add each text line as an overlay
  props.textLines.forEach((line) => {
    const lineId = line.line_id || line.text_block_id || `${line.x}_${line.y}`
    const isHighlighted = lineId === props.highlightedLineId

    const overlayDiv = document.createElement('div')
    overlayDiv.style.border = isHighlighted
      ? `4px solid ${getHighlightBorderColor()}`
      : `2px solid ${getSelectionBorderColor()}`
    overlayDiv.style.backgroundColor = isHighlighted
      ? getSelectionMediumColor()
      : getSelectionLightColor()
    overlayDiv.style.cursor = 'pointer'
    overlayDiv.style.boxSizing = 'border-box'
    overlayDiv.style.transition = 'all 0.2s'
    overlayDiv.style.borderRadius = '4px'
    overlayDiv.dataset.lineId = lineId

    // Add hover effect
    overlayDiv.addEventListener('mouseenter', () => {
      // Cancel any pending hover timeout
      if (hoverTimeout) {
        clearTimeout(hoverTimeout)
        hoverTimeout = null
      }
      overlayDiv.style.backgroundColor = getSelectionMediumColor()
      overlayDiv.style.border = `4px solid ${getHighlightBorderColor()}`
      emit('lineHover', lineId)
    })

    overlayDiv.addEventListener('mouseleave', () => {
      // Delay clearing the hover to prevent flash when moving between lines
      hoverTimeout = setTimeout(() => {
        if (lineId !== props.highlightedLineId) {
          overlayDiv.style.backgroundColor = getSelectionLightColor()
          overlayDiv.style.border = `2px solid ${getSelectionBorderColor()}`
        }
        emit('lineHover', null)
        hoverTimeout = null
      }, 50)  // 50ms delay is enough to bridge the gap between lines
    })

    // Click to select line
    overlayDiv.addEventListener('click', (e) => {
      e.stopPropagation()
      emit('lineClick', lineId)
    })

    // Normalize coordinates from ALTO space to OpenSeadragon normalized space [0,1]
    // ALTO coordinates are in high-res image space, need to normalize by ALTO dimensions
    const x = line.x / altoImageWidth
    const y = line.y / altoImageWidth  // Note: OSD uses width for both x and y normalization
    const width = line.width / altoImageWidth
    const height = line.height / altoImageWidth

    // Add overlay to viewer
    viewer!.addOverlay({
      element: overlayDiv,
      location: new OpenSeadragon.Rect(x, y, width, height),
    })

    textLineOverlays.set(lineId, overlayDiv)
  })
}

function addDetectionOverlays() {
  if (!viewer) return

  // Clear existing detection overlays
  detectionOverlays.forEach((element) => {
    viewer?.removeOverlay(element)
  })
  detectionOverlays.clear()

  // If no detections to show, just return after clearing
  if (!props.detections || props.detections.length === 0) {
    return
  }

  // Get image size from OpenSeadragon's world
  const tiledImage = viewer.world.getItemAt(0)
  if (!tiledImage) {
    return
  }

  // Get the actual image dimensions from OpenSeadragon
  const imageSize = tiledImage.getContentSize()
  const imageWidth = imageSize.x
  const imageHeight = imageSize.y

  if (!imageWidth || !imageHeight) {
    return
  }

  // Add each detection as an overlay
  props.detections.forEach((detection, index) => {
    // Check if bbox exists and has the right structure
    if (!detection.bbox) {
      console.error(`Detection ${index} missing bbox:`, detection)
      return
    }

    const color = getDetectionColor(detection.class_name)

    // Create overlay element
    const overlayDiv = document.createElement('div')
    overlayDiv.style.border = `3px solid ${color}`
    overlayDiv.style.backgroundColor = `${color}33`
    overlayDiv.style.pointerEvents = 'none'
    overlayDiv.style.boxSizing = 'border-box'

    // Create label
    const label = document.createElement('div')
    label.textContent = `${detection.class_name} (${(detection.confidence * 100).toFixed(0)}%)`
    label.style.position = 'absolute'
    label.style.top = '2px'
    label.style.left = '2px'
    label.style.backgroundColor = color
    label.style.color = 'white'
    label.style.padding = '2px 6px'
    label.style.fontSize = '12px'
    label.style.fontWeight = 'bold'
    label.style.borderRadius = '3px'
    label.style.whiteSpace = 'nowrap'
    overlayDiv.appendChild(label)

    // OpenSeadragon overlay coordinates for simple images:
    // When using type: 'image', OpenSeadragon treats the image as having dimensions (1, aspect_ratio)
    // So we need to normalize to image width = 1.0
    const x = detection.bbox.x1 / imageWidth
    const y = detection.bbox.y1 / imageWidth  // Note: normalize by width, not height!
    const width = (detection.bbox.x2 - detection.bbox.x1) / imageWidth
    const height = (detection.bbox.y2 - detection.bbox.y1) / imageWidth  // Note: normalize by width, not height!

    // Add overlay to viewer using normalized image coordinates
    if (viewer) {
      viewer.addOverlay({
        element: overlayDiv,
        location: new OpenSeadragon.Rect(x, y, width, height),
      })

      // Track the overlay for later removal
      detectionOverlays.set(detection.detection_id, overlayDiv)
    }
  })
}

onMounted(() => {
  if (viewerContainer.value) {
    viewer = OpenSeadragon({
      element: viewerContainer.value,
      prefixUrl: 'https://cdn.jsdelivr.net/npm/openseadragon@4.1/build/openseadragon/images/',
      tileSources: {
        type: 'image',
        url: props.imageUrl,
      },
      showNavigationControl: false, // We'll use custom controls
      showNavigator: props.showNavigator,
      navigatorPosition: props.navigatorPosition,
      navigatorHeight: props.navigatorHeight,
      navigatorWidth: props.navigatorWidth,
      gestureSettingsMouse: {
        clickToZoom: false,
        dblClickToZoom: true,
      },
      minZoomImageRatio: 0.8,
      maxZoomPixelRatio: 10,
      visibilityRatio: 0.2,
      constrainDuringPan: false,
      animationTime: 0.5,
    })

    // Add overlays when the image is fully loaded
    viewer.addHandler('open', () => {
      // Wait for the tiled image to be fully loaded
      const tiledImage = viewer!.world.getItemAt(0)
      if (tiledImage) {
        tiledImage.addHandler('fully-loaded-change', () => {
          addDetectionOverlays()
          addTextLineOverlays()
        })
        // If already loaded, add immediately
        if (tiledImage.getFullyLoaded()) {
          addDetectionOverlays()
          addTextLineOverlays()
        }
      }
    })
  }
})

onUnmounted(() => {
  if (viewer) {
    viewer.destroy()
    viewer = null
  }
})

watch(() => props.imageUrl, (newUrl) => {
  if (viewer) {
    viewer.open({
      type: 'image',
      url: newUrl,
    })
  }
})

watch(() => props.detections, () => {
  addDetectionOverlays()
}, { deep: true })

watch(() => props.textLines, () => {
  addTextLineOverlays()
}, { deep: true })

watch(() => props.highlightedLineId, (newId, oldId) => {
  // Update highlighted line styling when overlays are visible
  if (oldId) {
    const oldOverlay = textLineOverlays.get(oldId)
    if (oldOverlay) {
      oldOverlay.style.backgroundColor = getSelectionLightColor()
      oldOverlay.style.border = `2px solid ${getSelectionBorderColor()}`
    }
  }

  if (newId) {
    const newOverlay = textLineOverlays.get(newId)
    if (newOverlay) {
      newOverlay.style.backgroundColor = getSelectionMediumColor()
      newOverlay.style.border = `4px solid ${getHighlightBorderColor()}`
    }
  }

  // Handle single highlighted line overlay when text overlays are off
  updateHighlightedLineOverlay()
})

function updateHighlightedLineOverlay() {
  if (!viewer) return

  // Only show spotlight if textLines is empty but we have a highlighted line
  const shouldShow = (!props.textLines || props.textLines.length === 0)
    && props.highlightedLineId
    && props.allTextLines
    && props.allTextLines.length > 0

  // If we shouldn't show, clean up and return
  if (!shouldShow) {
    if (highlightedLineOverlay) {
      viewer.removeOverlay(highlightedLineOverlay)
      highlightedLineOverlay = null
    }
    if (spotlightOverlay) {
      spotlightOverlay.remove()
      spotlightOverlay = null
    }
    return
  }

  // Find the highlighted line in allTextLines
  const line = props.allTextLines.find(l => {
    const lineId = l.line_id || l.text_block_id || `${l.x}_${l.y}`
    return lineId === props.highlightedLineId
  })
  if (!line) return

  const tiledImage = viewer.world.getItemAt(0)
  if (!tiledImage) return

  const imageSize = tiledImage.getContentSize()
  const osdImageWidth = imageSize.x
  if (!osdImageWidth) return

  const altoImageWidth = props.imageWidth || osdImageWidth
  const container = viewerContainer.value
  if (!container) return

  // Calculate normalized coordinates
  const x = line.x / altoImageWidth
  const y = line.y / altoImageWidth
  const width = line.width / altoImageWidth
  const height = line.height / altoImageWidth

  // Helper to calculate clip-path
  const calcClipPath = () => {
    if (!viewer) return ''
    const rect = viewer.viewport.viewportToViewerElementRectangle(
      new OpenSeadragon.Rect(x, y, width, height)
    )
    const padding = 4
    const radius = 4  // Border radius for rounded corners
    const l = Math.max(0, rect.x - padding)
    const t = Math.max(0, rect.y - padding)
    const r = Math.min(container.clientWidth, rect.x + rect.width + padding)
    const b = Math.min(container.clientHeight, rect.y + rect.height + padding)

    // Create a rounded rectangle cutout using multiple points
    // The cutout needs to trace around a rounded rectangle
    return `polygon(
      evenodd,
      0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%,
      ${l + radius}px ${t}px,
      ${r - radius}px ${t}px,
      ${r}px ${t + radius}px,
      ${r}px ${b - radius}px,
      ${r - radius}px ${b}px,
      ${l + radius}px ${b}px,
      ${l}px ${b - radius}px,
      ${l}px ${t + radius}px,
      ${l + radius}px ${t}px
    )`
  }

  // Create or update spotlight overlay
  if (!spotlightOverlay) {
    spotlightOverlay = document.createElement('div')
    spotlightOverlay.className = 'spotlight-overlay'
    spotlightOverlay.style.cssText = `
      position: absolute;
      inset: 0;
      background: rgba(0, 0, 0, 0.4);
      pointer-events: none;
      z-index: 5;
    `
    container.appendChild(spotlightOverlay)

    // Update clip-path on viewport changes
    const updateClipPath = () => {
      if (spotlightOverlay) {
        spotlightOverlay.style.clipPath = calcClipPath()
      }
    }
    viewer.addHandler('animation', updateClipPath)
    viewer.addHandler('animation-finish', updateClipPath)
  }

  // Update clip-path immediately
  spotlightOverlay.style.clipPath = calcClipPath()
}

function zoomIn() {
  if (viewer) {
    viewer.viewport.zoomBy(1.3)
    viewer.viewport.applyConstraints()
  }
}

function zoomOut() {
  if (viewer) {
    viewer.viewport.zoomBy(0.7)
    viewer.viewport.applyConstraints()
  }
}

function resetZoom() {
  if (viewer) {
    viewer.viewport.goHome()
  }
}

function toggleFullscreen() {
  if (viewer) {
    viewer.setFullScreen(!viewer.isFullPage())
  }
}
</script>

<template>
  <div class="relative w-full h-full group">
    <!-- OpenSeadragon Container -->
    <div ref="viewerContainer" class="w-full h-full bg-black"></div>

    <!-- Custom Zoom Controls (horizontal, top right, autohide) -->
    <div v-if="showZoomControls" class="absolute top-4 right-4 flex flex-row gap-3 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        @click="zoomIn"
        class="icon-mask"
        title="Zoom In"
      >
        <ZoomIn class="h-6 w-6 stroke-[2.5]" />
      </button>
      <button
        @click="zoomOut"
        class="icon-mask"
        title="Zoom Out"
      >
        <ZoomOut class="h-6 w-6 stroke-[2.5]" />
      </button>
      <button
        @click="resetZoom"
        class="icon-mask"
        title="Reset Zoom"
      >
        <Home class="h-6 w-6 stroke-[2.5]" />
      </button>
      <button
        @click="toggleFullscreen"
        class="icon-mask"
        title="Fullscreen"
      >
        <Maximize2 class="h-6 w-6 stroke-[2.5]" />
      </button>
    </div>

    <!-- Left Navigation Overlay -->
    <button
      v-if="currentPage > 1"
      @click="emit('changePage', -1)"
      class="icon-mask absolute left-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-all z-10 p-1"
      title="Previous page"
    >
      <ChevronLeft class="h-14 w-14 stroke-[2.5]" />
    </button>

    <!-- Right Navigation Overlay -->
    <button
      v-if="currentPage < totalPages"
      @click="emit('changePage', 1)"
      class="icon-mask absolute right-0 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-all z-10 p-1"
      title="Next page"
    >
      <ChevronRight class="h-14 w-14 stroke-[2.5]" />
    </button>
  </div>
</template>

<style scoped>
/* Ensure OpenSeadragon canvas fills container */
:deep(.openseadragon-container) {
  width: 100% !important;
  height: 100% !important;
}

/* Make the navigator (minimap) display box white */
:deep(.displayregion) {
  border: 2px solid white !important;
}

/* Icon with soft drop shadow for visibility on any background */
.icon-mask svg {
  color: white;
  filter:
    drop-shadow(0 0 2px rgba(0, 0, 0, 0.5))
    drop-shadow(0 0 5px rgba(0, 0, 0, 0.4))
    drop-shadow(0 1px 2px rgba(0, 0, 0, 0.5));
}
</style>
