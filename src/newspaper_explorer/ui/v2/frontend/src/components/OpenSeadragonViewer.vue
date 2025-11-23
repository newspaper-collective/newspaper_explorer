<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import OpenSeadragon from 'openseadragon'
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Maximize2 } from 'lucide-vue-next'
import { getDetectionColor } from '@/lib/imageAnnotation'

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

function addTextLineOverlays() {
  if (!viewer) return
  
  // Clear existing text line overlays
  textLineOverlays.forEach((element) => {
    viewer?.removeOverlay(element)
  })
  textLineOverlays.clear()

  // If no text lines to show, just return after clearing
  if (!props.textLines || props.textLines.length === 0) {
    console.log('No text lines to display, overlays cleared')
    return
  }
  
  const tiledImage = viewer.world.getItemAt(0)
  if (!tiledImage) {
    console.warn('No tiled image loaded yet, skipping text line overlays')
    return
  }

  const imageSize = tiledImage.getContentSize()
  const osdImageWidth = imageSize.x
  const osdImageHeight = imageSize.y

  if (!osdImageWidth || !osdImageHeight) {
    console.warn('Image dimensions not available, skipping text line overlays')
    return
  }

  // Use provided ALTO image dimensions if available, otherwise fall back to OSD dimensions
  const altoImageWidth = props.imageWidth || osdImageWidth
  const altoImageHeight = props.imageHeight || osdImageHeight

  console.log(`Adding ${props.textLines.length} text line overlays (ALTO dims: ${altoImageWidth}x${altoImageHeight}, OSD dims: ${osdImageWidth}x${osdImageHeight})`)

  // Add each text line as an overlay
  props.textLines.forEach((line) => {
    const lineId = line.line_id || line.text_block_id || `${line.x}_${line.y}`
    const isHighlighted = lineId === props.highlightedLineId
    
    const overlayDiv = document.createElement('div')
    overlayDiv.style.border = isHighlighted ? '4px solid #2563eb' : '2px solid rgba(59, 130, 246, 0.5)'
    overlayDiv.style.backgroundColor = isHighlighted ? 'rgba(37, 99, 235, 0.4)' : 'rgba(59, 130, 246, 0.1)'
    overlayDiv.style.cursor = 'pointer'
    overlayDiv.style.boxSizing = 'border-box'
    overlayDiv.style.transition = 'all 0.2s'
    overlayDiv.dataset.lineId = lineId
    
    // Add hover effect
    overlayDiv.addEventListener('mouseenter', () => {
      overlayDiv.style.backgroundColor = 'rgba(37, 99, 235, 0.4)'
      overlayDiv.style.border = '4px solid #2563eb'
      emit('lineHover', lineId)
    })
    
    overlayDiv.addEventListener('mouseleave', () => {
      if (lineId !== props.highlightedLineId) {
        overlayDiv.style.backgroundColor = 'rgba(59, 130, 246, 0.1)'
        overlayDiv.style.border = '2px solid rgba(59, 130, 246, 0.5)'
      }
      emit('lineHover', null)
    })
    
    // Click to select line
    overlayDiv.addEventListener('click', (e) => {
      console.log('Overlay clicked in OpenSeadragonViewer:', lineId)
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
    console.log('No detections to display, overlays cleared')
    return
  }
  
  // Get image size from OpenSeadragon's world
  const tiledImage = viewer.world.getItemAt(0)
  if (!tiledImage) {
    console.warn('No tiled image loaded yet, skipping overlay rendering')
    return
  }

  // Get the actual image dimensions from OpenSeadragon
  const imageSize = tiledImage.getContentSize()
  const imageWidth = imageSize.x
  const imageHeight = imageSize.y

  if (!imageWidth || !imageHeight) {
    console.warn('Image dimensions not available from OpenSeadragon, skipping overlay rendering')
    return
  }

  console.log(`Adding ${props.detections.length} detection overlays to OpenSeadragon image ${imageWidth}x${imageHeight}`)
  console.log('First detection sample:', JSON.stringify(props.detections[0], null, 2))

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

    if (index < 3) {  // Log first 3 detections in detail
      console.log(`Detection ${detection.class_name}: pixel coords (${detection.bbox.x1.toFixed(0)}, ${detection.bbox.y1.toFixed(0)}, ${detection.bbox.x2.toFixed(0)}, ${detection.bbox.y2.toFixed(0)}) -> normalized (${x.toFixed(4)}, ${y.toFixed(4)}, ${width.toFixed(4)}, ${height.toFixed(4)})`)
      console.log(`Image dimensions: ${imageWidth}x${imageHeight}, aspect ratio: ${(imageHeight/imageWidth).toFixed(4)}`)
    }

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
      console.log('OpenSeadragon: Image opened')
      // Wait for the tiled image to be fully loaded
      const tiledImage = viewer!.world.getItemAt(0)
      if (tiledImage) {
        tiledImage.addHandler('fully-loaded-change', () => {
          console.log('OpenSeadragon: Image fully loaded, adding overlays')
          addDetectionOverlays()
          addTextLineOverlays()
        })
        // If already loaded, add immediately
        if (tiledImage.getFullyLoaded()) {
          console.log('OpenSeadragon: Image already fully loaded, adding overlays')
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
    console.log('Image URL changed, opening new image:', newUrl)
    viewer.open({
      type: 'image',
      url: newUrl,
    })
  }
})

watch(() => props.detections, () => {
  console.log('Detections changed, re-adding overlays...')
  addDetectionOverlays()
}, { deep: true })

watch(() => props.textLines, () => {
  console.log('Text lines changed, re-adding overlays...')
  addTextLineOverlays()
}, { deep: true })

watch(() => props.highlightedLineId, (newId, oldId) => {
  // Update highlighted line styling
  if (oldId) {
    const oldOverlay = textLineOverlays.get(oldId)
    if (oldOverlay) {
      oldOverlay.style.backgroundColor = 'rgba(59, 130, 246, 0.1)'
      oldOverlay.style.border = '2px solid rgba(59, 130, 246, 0.5)'
    }
  }
  
  if (newId) {
    const newOverlay = textLineOverlays.get(newId)
    if (newOverlay) {
      newOverlay.style.backgroundColor = 'rgba(37, 99, 235, 0.4)'
      newOverlay.style.border = '4px solid #2563eb'
    }
  }
})

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
</script>

<template>
  <div class="relative w-full h-full group">
    <!-- OpenSeadragon Container -->
    <div ref="viewerContainer" class="w-full h-full bg-black"></div>

    <!-- Custom Zoom Controls (horizontal, left side, autohide) -->
    <div v-if="showZoomControls" class="absolute top-4 left-20 flex flex-row gap-2 z-10 opacity-0 group-hover:opacity-100 transition-opacity">
      <button
        @click="zoomIn"
        class="p-2 rounded-lg bg-background/90 hover:bg-background shadow-lg transition-colors"
        title="Zoom In"
      >
        <ZoomIn class="h-5 w-5" />
      </button>
      <button
        @click="zoomOut"
        class="p-2 rounded-lg bg-background/90 hover:bg-background shadow-lg transition-colors"
        title="Zoom Out"
      >
        <ZoomOut class="h-5 w-5" />
      </button>
      <button
        @click="resetZoom"
        class="p-2 rounded-lg bg-background/90 hover:bg-background shadow-lg transition-colors"
        title="Reset Zoom"
      >
        <Maximize2 class="h-5 w-5" />
      </button>
    </div>

    <!-- Left Navigation Overlay -->
    <button
      v-if="currentPage > 1"
      @click="emit('changePage', -1)"
      class="absolute left-0 top-0 h-full w-16 flex items-center justify-start pl-2 opacity-0 group-hover:opacity-100 hover:bg-black/20 transition-all z-10"
    >
      <div class="p-2 rounded-full bg-background/90 shadow-lg">
        <ChevronLeft class="h-6 w-6" />
      </div>
    </button>

    <!-- Right Navigation Overlay -->
    <button
      v-if="currentPage < totalPages"
      @click="emit('changePage', 1)"
      class="absolute right-0 top-0 h-full w-16 flex items-center justify-end pr-2 opacity-0 group-hover:opacity-100 hover:bg-black/20 transition-all z-10"
    >
      <div class="p-2 rounded-full bg-background/90 shadow-lg">
        <ChevronRight class="h-6 w-6" />
      </div>
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
</style>
