/**
 * Image utility functions for newspaper page and detection handling
 * 
 * Reusable across PicturesView, LayoutView, BrowseView, IssueView
 */

export interface PageMetadata {
  source: string
  date: string
  issue: string
  daily: string
  page: string
}

/**
 * Format date string to human-readable format
 */
export function formatDate(dateStr?: string): string {
  if (!dateStr) return 'Unknown date'
  try {
    const date = new Date(dateStr)
    return date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
    })
  } catch {
    return dateStr
  }
}

/**
 * Parse page_id into structured metadata
 * 
 * @param pageId Format: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}
 * @returns Parsed metadata or null if invalid format
 */
export function parsePageMetadata(pageId: string): PageMetadata | null {
  const parts = pageId.split('_')
  if (parts.length >= 5) {
    return {
      source: parts[0],
      date: parts[1],
      issue: parts[2],
      daily: parts[3],
      page: parts[4],
    }
  }
  return null
}

/**
 * Extract date from page_id (YYYY-MM-DD format)
 */
export function extractDateFromPageId(pageId: string): string | null {
  const dateMatch = pageId.match(/_(\d{4}-\d{2}-\d{2})_/)
  return dateMatch ? dateMatch[1] : null
}

/**
 * Construct full image URL from relative path
 * 
 * Handles both absolute and relative paths from backend
 */
export function getFullImageUrl(imagePath: string, source: string): string | null {
  if (!imagePath || !source) return null
  
  let relativePath = imagePath
  
  // If it's an absolute path, extract the part after /images/
  if (relativePath.includes('/images/')) {
    relativePath = relativePath.split('/images/')[1]
  }
  
  return `/static/${source}/images/${relativePath}`
}

/**
 * Format page metadata for display
 */
export function formatPageMetadataDisplay(metadata: PageMetadata): string {
  return `Issue ${metadata.issue} • Daily ${metadata.daily} • Page ${metadata.page}`
}
