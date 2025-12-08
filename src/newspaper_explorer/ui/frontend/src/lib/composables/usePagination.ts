/**
 * Generic pagination composable
 * 
 * Reusable pagination logic for server-side paginated data
 * Used in PicturesView, LayoutView, EntitiesView, TopicsView
 */

import { ref, computed } from 'vue'

export interface PaginationOptions {
  itemsPerPage?: number
  initialPage?: number
}

export function usePagination(options: PaginationOptions = {}) {
  const { itemsPerPage = 12, initialPage = 1 } = options

  const currentPage = ref(initialPage)
  const totalItems = ref(0)
  const pageSize = ref(itemsPerPage)

  const totalPages = computed(() => {
    return Math.ceil(totalItems.value / pageSize.value)
  })

  const hasNextPage = computed(() => {
    return currentPage.value < totalPages.value
  })

  const hasPreviousPage = computed(() => {
    return currentPage.value > 1
  })

  function goToPage(page: number) {
    if (page < 1 || page > totalPages.value) {
      return
    }
    currentPage.value = page
  }

  function nextPage() {
    if (hasNextPage.value) {
      currentPage.value++
    }
  }

  function previousPage() {
    if (hasPreviousPage.value) {
      currentPage.value--
    }
  }

  function reset() {
    currentPage.value = initialPage
    totalItems.value = 0
  }

  function setPageSize(size: number) {
    pageSize.value = size
    // Reset to page 1 when changing page size
    currentPage.value = 1
  }

  return {
    // State
    currentPage,
    totalItems,
    totalPages,
    pageSize,
    hasNextPage,
    hasPreviousPage,

    // Methods
    goToPage,
    nextPage,
    previousPage,
    reset,
    setPageSize,
  }
}
