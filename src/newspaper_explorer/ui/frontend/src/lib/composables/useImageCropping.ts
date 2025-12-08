/**
 * Image cropping composable using Canvas API
 * 
 * Handles client-side image cropping with caching for performance
 * Used in PicturesView, LayoutView for detection previews
 */

import { ref } from 'vue'

export interface BBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export function useImageCropping() {
  const croppedImages = ref<Record<string, string>>({})
  const croppedCaptions = ref<Record<string, string>>({})
  const cropLoadingImages = ref<Set<string>>(new Set())
  const cropLoadingCaptions = ref<Set<string>>(new Set())

  /**
   * Crop an image region and return as data URL
   * 
   * @param imageUrl Full image URL
   * @param bbox Bounding box to crop
   * @param cacheKey Unique cache key for this crop
   * @param maxWidth Maximum width for the cropped image (maintains aspect ratio)
   */
  async function loadCroppedImage(
    imageUrl: string,
    bbox: BBox,
    cacheKey: string,
    maxWidth: number = 400
  ): Promise<void> {
    // Skip if already loaded or currently loading
    if (croppedImages.value[cacheKey] || cropLoadingImages.value.has(cacheKey)) {
      return
    }

    cropLoadingImages.value.add(cacheKey)

    try {
      // Load the full image
      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('Failed to load image'))
        img.src = imageUrl
      })

      // Create canvas for cropping
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Failed to get canvas context')
      }

      // Calculate crop dimensions
      const cropWidth = bbox.x2 - bbox.x1
      const cropHeight = bbox.y2 - bbox.y1

      // Set canvas size with max width while maintaining aspect ratio
      const scale = Math.min(1, maxWidth / cropWidth)
      canvas.width = cropWidth * scale
      canvas.height = cropHeight * scale

      // Draw the cropped region
      ctx.drawImage(
        img,
        bbox.x1,
        bbox.y1,
        cropWidth,
        cropHeight, // source rectangle
        0,
        0,
        canvas.width,
        canvas.height // destination rectangle
      )

      // Convert to data URL and store in cache
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
      croppedImages.value[cacheKey] = dataUrl
    } catch (error) {
      console.error('Failed to crop image:', error)
    } finally {
      cropLoadingImages.value.delete(cacheKey)
    }
  }

  /**
   * Crop a caption region (typically smaller than pictures)
   * 
   * @param imageUrl Full image URL (same page as picture)
   * @param bbox Caption bounding box
   * @param cacheKey Unique cache key
   * @param maxWidth Maximum width (default 800px for captions)
   */
  async function loadCroppedCaption(
    imageUrl: string,
    bbox: BBox,
    cacheKey: string,
    maxWidth: number = 800
  ): Promise<void> {
    // Skip if already loaded or currently loading
    if (croppedCaptions.value[cacheKey] || cropLoadingCaptions.value.has(cacheKey)) {
      return
    }

    cropLoadingCaptions.value.add(cacheKey)

    try {
      // Load the full image
      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('Failed to load image'))
        img.src = imageUrl
      })

      // Create canvas for cropping caption
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Failed to get canvas context')
      }

      // Calculate crop dimensions for caption
      const cropWidth = bbox.x2 - bbox.x1
      const cropHeight = bbox.y2 - bbox.y1

      // Set canvas size (captions are usually small, use actual size)
      canvas.width = Math.min(cropWidth, maxWidth)
      canvas.height = cropHeight * (canvas.width / cropWidth)

      // Draw the cropped caption region
      ctx.drawImage(
        img,
        bbox.x1,
        bbox.y1,
        cropWidth,
        cropHeight,
        0,
        0,
        canvas.width,
        canvas.height
      )

      // Convert to data URL and store
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9)
      croppedCaptions.value[cacheKey] = dataUrl
    } catch (error) {
      console.error('Failed to crop caption:', error)
    } finally {
      cropLoadingCaptions.value.delete(cacheKey)
    }
  }

  /**
   * Create a page thumbnail with detection overlays
   * 
   * @param imageUrl Full page image URL
   * @param detections Array of detections to overlay
   * @param maxWidth Thumbnail width
   * @returns Data URL of thumbnail with overlays
   */
  async function createPageThumbnailWithOverlay(
    imageUrl: string,
    detections: Array<{ bbox: BBox; color: string }>,
    maxWidth: number = 300
  ): Promise<string | null> {
    try {
      // Load the full image
      const img = new Image()
      img.crossOrigin = 'anonymous'

      await new Promise<void>((resolve, reject) => {
        img.onload = () => resolve()
        img.onerror = () => reject(new Error('Failed to load image'))
        img.src = imageUrl
      })

      // Create canvas for thumbnail
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Failed to get canvas context')
      }

      // Scale to thumbnail size
      const scale = maxWidth / img.width
      canvas.width = maxWidth
      canvas.height = img.height * scale

      // Draw the full page
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Draw detection boxes
      detections.forEach(({ bbox, color }) => {
        ctx.strokeStyle = color
        ctx.lineWidth = 2
        ctx.strokeRect(
          bbox.x1 * scale,
          bbox.y1 * scale,
          (bbox.x2 - bbox.x1) * scale,
          (bbox.y2 - bbox.y1) * scale
        )
      })

      // Convert to data URL
      return canvas.toDataURL('image/jpeg', 0.85)
    } catch (error) {
      console.error('Failed to create page thumbnail:', error)
      return null
    }
  }

  /**
   * Clear all cached images
   */
  function clearCache() {
    croppedImages.value = {}
    croppedCaptions.value = {}
    cropLoadingImages.value.clear()
    cropLoadingCaptions.value.clear()
  }

  /**
   * Clear specific cache entries
   */
  function clearCacheEntry(cacheKey: string) {
    delete croppedImages.value[cacheKey]
    delete croppedCaptions.value[cacheKey]
    cropLoadingImages.value.delete(cacheKey)
    cropLoadingCaptions.value.delete(cacheKey)
  }

  return {
    // State
    croppedImages,
    croppedCaptions,
    cropLoadingImages,
    cropLoadingCaptions,

    // Methods
    loadCroppedImage,
    loadCroppedCaption,
    createPageThumbnailWithOverlay,
    clearCache,
    clearCacheEntry,
  }
}
