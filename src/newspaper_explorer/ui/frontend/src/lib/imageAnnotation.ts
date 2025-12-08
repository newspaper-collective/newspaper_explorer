/**
 * Image annotation utilities for drawing bounding boxes on images
 */

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export interface Detection {
  detection_id: string
  class_name: string
  confidence: number
  bbox: BoundingBox
  text_content?: string
}

export interface AnnotationOptions {
  maxWidth?: number
  lineWidth?: number
  fontSize?: number
  showLabels?: boolean
  colors?: Record<string, string>
}

// Color map for detection classes
export const DETECTION_COLORS: Record<string, string> = {
  'Text': '#FF4444',
  'Picture': '#44FF44',
  'Section-header': '#4444FF',
  'Table': '#FFFF44',
  'Page-header': '#FF44FF',
  'Page-footer': '#44FFFF',
  'Caption': '#FFA500',
  'List': '#800080',
  'Title': '#FF1493',
  'Figure': '#00CED1',
  'Formula': '#FFD700',
}

/**
 * Draw bounding boxes on a canvas
 */
export function drawAnnotations(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  detections: Detection[],
  options: AnnotationOptions = {}
): void {
  const {
    maxWidth = 800,
    lineWidth = 2,
    fontSize = 12,
    showLabels = true,
    colors = DETECTION_COLORS,
  } = options

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  // Calculate scale
  let scale = 1
  if (image.width > maxWidth) {
    scale = maxWidth / image.width
  }

  const displayWidth = image.width * scale
  const displayHeight = image.height * scale

  // Set canvas size
  canvas.width = displayWidth
  canvas.height = displayHeight

  // Draw image
  ctx.drawImage(image, 0, 0, displayWidth, displayHeight)

  // Draw each detection
  for (const detection of detections) {
    const { bbox, class_name, confidence } = detection
    const color = colors[class_name] || '#FFFFFF'

    // Scale coordinates
    const x1 = bbox.x1 * scale
    const y1 = bbox.y1 * scale
    const x2 = bbox.x2 * scale
    const y2 = bbox.y2 * scale

    // Draw rectangle
    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

    // Draw label if enabled
    if (showLabels) {
      const label = `${class_name} ${confidence.toFixed(2)}`
      ctx.font = `${fontSize}px sans-serif`

      // Measure text
      const metrics = ctx.measureText(label)
      const textWidth = metrics.width
      const textHeight = fontSize

      // Draw label background
      ctx.fillStyle = color
      ctx.fillRect(x1, y1 - textHeight - 4, textWidth + 4, textHeight + 4)

      // Draw label text
      ctx.fillStyle = 'black'
      ctx.fillText(label, x1 + 2, y1 - 4)
    }
  }
}

/**
 * Create an annotated image as a data URL
 */
export async function createAnnotatedImageUrl(
  imageUrl: string,
  detections: Detection[],
  options: AnnotationOptions = {}
): Promise<string> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.crossOrigin = 'anonymous'

    image.onload = () => {
      const canvas = document.createElement('canvas')
      drawAnnotations(canvas, image, detections, options)
      resolve(canvas.toDataURL('image/jpeg', 0.85))
    }

    image.onerror = () => {
      reject(new Error(`Failed to load image: ${imageUrl}`))
    }

    image.src = imageUrl
  })
}

/**
 * Get color for a detection class
 */
export function getDetectionColor(className: string): string {
  return DETECTION_COLORS[className] || '#FFFFFF'
}
