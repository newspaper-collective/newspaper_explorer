/**
 * Keyboard navigation composable for dialogs and galleries
 * 
 * Handles arrow key navigation with cross-page support
 * Used in PicturesView, IssueView, LayoutView dialogs
 */

import { ref, computed, watch, onMounted, onUnmounted, type Ref } from 'vue'

export interface NavigationItem {
  id: string
  [key: string]: any
}

export interface NavigationOptions<T extends NavigationItem> {
  items: Ref<T[]>
  currentItem: Ref<T | null>
  currentPage?: Ref<number>
  totalPages?: Ref<number>
  onNavigate: (item: T) => void
  onPageChange?: (page: number) => void
  enabled?: Ref<boolean>
}

export function useKeyboardNavigation<T extends NavigationItem>(
  options: NavigationOptions<T>
) {
  const { items, currentItem, currentPage, totalPages, onNavigate, onPageChange, enabled } = options

  const currentIndex = computed(() => {
    if (!currentItem.value) return -1
    return items.value.findIndex((item) => item.id === currentItem.value?.id)
  })

  const hasPrevious = computed(() => {
    if (currentPage && currentPage.value > 1) return true
    return currentIndex.value > 0
  })

  const hasNext = computed(() => {
    const isLastOnPage = currentIndex.value === items.value.length - 1
    if (totalPages && currentPage && currentPage.value < totalPages.value) {
      return true
    }
    return !isLastOnPage
  })

  function previous() {
    if (currentIndex.value > 0) {
      // Normal case: previous item is on current page
      const prev = items.value[currentIndex.value - 1]
      onNavigate(prev)
    } else if (currentPage && totalPages && onPageChange && currentPage.value > 1) {
      // Cross-page case: previous item is on previous page
      currentPage.value--

      // Wait for items to update
      const unwatch = watch(items, (newItems) => {
        if (newItems.length > 0) {
          // Navigate to LAST item of the previous page
          onNavigate(newItems[newItems.length - 1])
          unwatch()
        }
      })
    }
  }

  function next() {
    const isLastOnPage = currentIndex.value === items.value.length - 1

    if (!isLastOnPage && currentIndex.value !== -1) {
      // Normal case: next item is on current page
      const nextItem = items.value[currentIndex.value + 1]
      onNavigate(nextItem)
    } else if (currentPage && totalPages && onPageChange && currentPage.value < totalPages.value) {
      // Cross-page case: next item is on next page
      currentPage.value++

      // Wait for items to update
      const unwatch = watch(items, (newItems) => {
        if (newItems.length > 0) {
          // Navigate to FIRST item of the next page
          onNavigate(newItems[0])
          unwatch()
        }
      })
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    // Only handle if enabled (dialog is open, etc.)
    if (enabled && !enabled.value) return

    if (e.key === 'ArrowLeft') {
      e.preventDefault()
      previous()
    } else if (e.key === 'ArrowRight') {
      e.preventDefault()
      next()
    } else if (e.key === 'Escape' && enabled) {
      e.preventDefault()
      // Let parent handle closing
    }
  }

  onMounted(() => {
    window.addEventListener('keydown', handleKeydown)
  })

  onUnmounted(() => {
    window.removeEventListener('keydown', handleKeydown)
  })

  return {
    // Computed
    currentIndex,
    hasPrevious,
    hasNext,

    // Methods
    previous,
    next,
    handleKeydown,
  }
}
