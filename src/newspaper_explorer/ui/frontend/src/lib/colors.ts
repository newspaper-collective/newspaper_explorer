/**
 * Centralized color palette utilities
 *
 * All analysis colors are defined as CSS variables in index.css
 * This module provides helper functions to access them consistently
 */

// Cache for computed colors (cleared on theme change)
let colorCache: Map<string, string> = new Map()

// Listen for theme changes to clear cache
if (typeof window !== 'undefined') {
  const observer = new MutationObserver(() => {
    colorCache.clear()
  })
  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['class'],
  })
}

/**
 * Get a CSS variable value and convert HSL to hex for chart libraries
 */
function getCssColor(varName: string): string {
  if (typeof window === 'undefined') return '#888888'

  // Check cache first
  const cached = colorCache.get(varName)
  if (cached) return cached

  const hsl = getComputedStyle(document.documentElement)
    .getPropertyValue(varName)
    .trim()

  if (!hsl) return '#888888'

  // Parse HSL values (format: "224 100% 59%" or "224 100% 59% / 0.5")
  const parts = hsl.split('/').map(s => s.trim())
  const [h, s, l] = parts[0].split(' ').map((v) => parseFloat(v))
  const alpha = parts[1] ? parseFloat(parts[1]) : 1

  const hex = alpha < 1 ? hslToRgba(h, s, l, alpha) : hslToHex(h, s, l)
  colorCache.set(varName, hex)
  return hex
}

/**
 * Convert HSL to Hex color
 */
function hslToHex(h: number, s: number, l: number): string {
  s /= 100
  l /= 100

  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = l - c / 2

  let r = 0,
    g = 0,
    b = 0

  if (h < 60) {
    r = c; g = x
  } else if (h < 120) {
    r = x; g = c
  } else if (h < 180) {
    g = c; b = x
  } else if (h < 240) {
    g = x; b = c
  } else if (h < 300) {
    r = x; b = c
  } else {
    r = c; b = x
  }

  const toHex = (n: number) =>
    Math.round((n + m) * 255)
      .toString(16)
      .padStart(2, '0')

  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

/**
 * Convert HSL to RGBA string
 */
function hslToRgba(h: number, s: number, l: number, a: number): string {
  s /= 100
  l /= 100

  const c = (1 - Math.abs(2 * l - 1)) * s
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1))
  const m = l - c / 2

  let r = 0, g = 0, b = 0

  if (h < 60) {
    r = c; g = x
  } else if (h < 120) {
    r = x; g = c
  } else if (h < 180) {
    g = c; b = x
  } else if (h < 240) {
    g = x; b = c
  } else if (h < 300) {
    r = x; b = c
  } else {
    r = c; b = x
  }

  const toRgb = (n: number) => Math.round((n + m) * 255)
  return `rgba(${toRgb(r)}, ${toRgb(g)}, ${toRgb(b)}, ${a})`
}

// ===== Result Palette (base chart colors) =====

export const RESULT_COUNT = 12

export function getResultColor(index: number): string {
  const i = ((index - 1) % 12) + 1 // 1-indexed, wraps around
  return getCssColor(`--result-${i}`)
}

export function getResultColors(count: number = 12): string[] {
  return Array.from({ length: count }, (_, i) => getResultColor(i + 1))
}

// ===== Entity Type Colors =====

export const ENTITY_TYPES = ['person', 'organization', 'location', 'date', 'event', 'misc'] as const
export type EntityType = (typeof ENTITY_TYPES)[number]

export function getEntityColor(entityType: string): string {
  const type = entityType.toLowerCase()
  return getCssColor(`--entity-${type}`)
}

export function getEntityTypeColors(): Record<string, string> {
  return Object.fromEntries(ENTITY_TYPES.map((type) => [type, getEntityColor(type)]))
}

// ===== Emotion Colors =====

export const EMOTIONS = ['joy', 'love', 'anger', 'fear', 'sadness', 'agitation'] as const
export type Emotion = (typeof EMOTIONS)[number]

export function getEmotionColor(emotion: string): string {
  const type = emotion.toLowerCase()
  return getCssColor(`--emotion-${type}`)
}

export function getEmotionColors(): Record<string, string> {
  return Object.fromEntries(EMOTIONS.map((emo) => [emo, getEmotionColor(emo)]))
}

// ===== Layout/Detection Class Colors =====

export const LAYOUT_CLASSES = [
  'Text',
  'Picture',
  'Section-header',
  'Table',
  'Page-header',
  'Page-footer',
  'Caption',
  'List',
  'Title',
  'Figure',
  'Formula',
] as const
export type LayoutClass = (typeof LAYOUT_CLASSES)[number]

export function getLayoutColor(className: string): string {
  const varKey = className.toLowerCase()
  return getCssColor(`--layout-${varKey}`)
}

export function getLayoutColors(): Record<string, string> {
  return Object.fromEntries(LAYOUT_CLASSES.map((cls) => [cls, getLayoutColor(cls)]))
}

// Alias for backward compatibility
export const getDetectionColor = getLayoutColor
export const DETECTION_COLORS = new Proxy({} as Record<string, string>, {
  get(_, prop: string) {
    return getLayoutColor(prop)
  },
})

// ===== Preprocessing Category Colors =====

export const PREPROCESSING_CATEGORIES = [
  'normalization',
  'cleaning',
  'filtering',
  'modernization',
  'linguistic',
  'quality',
] as const
export type PreprocessingCategory = (typeof PREPROCESSING_CATEGORIES)[number]

export function getPreprocessingColor(category: string): string {
  const key = category.toLowerCase()
  return getCssColor(`--preprocessing-${key}`)
}

export function getPreprocessingColors(): Record<string, string> {
  return Object.fromEntries(
    PREPROCESSING_CATEGORIES.map((cat) => [cat, getPreprocessingColor(cat)])
  )
}

// ===== Selection/Highlight Colors =====

export function getSelectionColor(): string {
  return getCssColor('--selection')
}

export function getSelectionLightColor(): string {
  return getCssColor('--selection-light')
}

export function getSelectionMediumColor(): string {
  return getCssColor('--selection-medium')
}

export function getSelectionBorderColor(): string {
  return getCssColor('--selection-border')
}

export function getHighlightColor(): string {
  return getCssColor('--highlight')
}

export function getHighlightBorderColor(): string {
  return getCssColor('--highlight-border')
}

// ===== Overlay Colors =====

export function getOverlayColor(strength: 'light' | 'medium' | 'heavy' = 'light'): string {
  return getCssColor(`--overlay-${strength}`)
}

// ===== Chart Theme Colors =====

export function getChartTooltipBg(): string {
  return getCssColor('--chart-tooltip-bg')
}

export function getChartTooltipBorder(): string {
  return getCssColor('--chart-tooltip-border')
}

export function getChartTooltipText(): string {
  return getCssColor('--chart-tooltip-text')
}

export function getChartGridColor(): string {
  return getCssColor('--chart-grid')
}

export function getChartAxisPointerColor(): string {
  return getCssColor('--chart-axis-pointer')
}

export function getChartShadowColor(): string {
  return getCssColor('--chart-shadow')
}

export function getChartIconBorderColor(): string {
  return getCssColor('--chart-icon-border')
}

export function getChartPieBorderColor(): string {
  return getCssColor('--chart-pie-border')
}

export function getChartTextShadowColor(): string {
  return getCssColor('--chart-text-shadow')
}

// ===== Convenience: Full Chart Theme =====

export function getChartThemeColors() {
  return {
    palette: getResultColors(12),
    tooltip: {
      backgroundColor: getChartTooltipBg(),
      borderColor: getChartTooltipBorder(),
      textColor: getChartTooltipText(),
    },
    grid: getChartGridColor(),
    axisPointer: getChartAxisPointerColor(),
  }
}
