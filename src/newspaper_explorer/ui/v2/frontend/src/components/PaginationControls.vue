<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-vue-next'

interface Props {
  currentPage: number
  totalPages: number
  loading?: boolean
}

interface Emits {
  (e: 'update:currentPage', page: number): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

const menuOpen = ref(false)
const inputValue = ref('')
const menuRef = ref<HTMLElement | null>(null)

const goToPage = (page: number) => {
  if (page < 1 || page > props.totalPages || props.loading) {
    return
  }
  emit('update:currentPage', page)
  menuOpen.value = false
}

const previousPage = () => {
  if (props.currentPage > 1) {
    goToPage(props.currentPage - 1)
  }
}

const nextPage = () => {
  if (props.currentPage < props.totalPages) {
    goToPage(props.currentPage + 1)
  }
}

const firstPage = () => {
  goToPage(1)
}

const lastPage = () => {
  goToPage(props.totalPages)
}

const toggleMenu = () => {
  menuOpen.value = !menuOpen.value
  if (menuOpen.value) {
    inputValue.value = props.currentPage.toString()
  }
}

const handleInputSubmit = () => {
  const page = parseInt(inputValue.value)
  if (!isNaN(page)) {
    goToPage(page)
  }
}

// Generate page list with smart truncation
const pageList = computed(() => {
  const pages: (number | string)[] = []
  const total = props.totalPages
  const current = props.currentPage
  
  if (total <= 10) {
    // Show all pages if 10 or fewer
    for (let i = 1; i <= total; i++) {
      pages.push(i)
    }
  } else {
    // Always show first page
    pages.push(1)
    
    // Show pages around current page
    const start = Math.max(2, current - 2)
    const end = Math.min(total - 1, current + 2)
    
    if (start > 2) {
      pages.push('...')
    }
    
    for (let i = start; i <= end; i++) {
      pages.push(i)
    }
    
    if (end < total - 1) {
      pages.push('...')
    }
    
    // Always show last page
    pages.push(total)
  }
  
  return pages
})

// Close menu when clicking outside
const handleClickOutside = (event: MouseEvent) => {
  if (menuRef.value && !menuRef.value.contains(event.target as Node)) {
    menuOpen.value = false
  }
}

watch(menuOpen, (isOpen) => {
  if (isOpen) {
    document.addEventListener('click', handleClickOutside)
  } else {
    document.removeEventListener('click', handleClickOutside)
  }
})
</script>

<template>
  <div v-if="totalPages > 1" class="flex items-center justify-center gap-2 pt-4">
    <!-- First page -->
    <button
      @click="firstPage"
      :disabled="currentPage === 1 || loading"
      class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      title="First page"
    >
      <ChevronsLeft class="h-5 w-5" />
    </button>
    
    <!-- Previous page -->
    <button
      @click="previousPage"
      :disabled="currentPage === 1 || loading"
      class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      title="Previous page"
    >
      <ChevronLeft class="h-5 w-5" />
    </button>
    
    <!-- Page selector with dropdown -->
    <div class="relative" ref="menuRef">
      <button
        @click="toggleMenu"
        :disabled="loading"
        class="text-sm font-medium min-w-[80px] px-3 py-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        title="Select page"
      >
        {{ currentPage }} / {{ totalPages }}
      </button>
      
      <!-- Dropdown menu -->
      <div
        v-if="menuOpen"
        class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-popover border rounded-lg shadow-lg p-3 z-50 min-w-[200px]"
      >
        <!-- Direct input -->
        <div class="mb-3">
          <label class="text-xs text-muted-foreground block mb-1">Go to page:</label>
          <form @submit.prevent="handleInputSubmit" class="flex">
            <input
              v-model="inputValue"
              type="number"
              :min="1"
              :max="totalPages"
              class="flex-1 px-3 py-1.5 text-sm border border-r-0 rounded-l-md bg-background focus:outline-none focus:ring-2 focus:ring-ring focus:z-10"
              placeholder="Page #"
              @click.stop
            />
            <button
              type="submit"
              class="px-4 py-1.5 text-sm bg-primary text-primary-foreground rounded-r-md hover:bg-primary/90 transition-colors border border-primary"
            >
              Go
            </button>
          </form>
        </div>
        
        <!-- Page list -->
        <div class="border-t pt-2">
          <div class="text-xs text-muted-foreground mb-1">Select page:</div>
          <div class="max-h-[200px] overflow-y-auto space-y-1">
            <button
              v-for="page in pageList"
              :key="page"
              @click="typeof page === 'number' ? goToPage(page) : undefined"
              :disabled="page === '...'"
              :class="[
                'w-full text-left px-2 py-1 text-sm rounded transition-colors',
                page === currentPage
                  ? 'bg-accent font-semibold'
                  : page === '...'
                  ? 'cursor-default text-muted-foreground'
                  : 'hover:bg-accent'
              ]"
            >
              {{ page }}
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Next page -->
    <button
      @click="nextPage"
      :disabled="currentPage === totalPages || loading"
      class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      title="Next page"
    >
      <ChevronRight class="h-5 w-5" />
    </button>
    
    <!-- Last page -->
    <button
      @click="lastPage"
      :disabled="currentPage === totalPages || loading"
      class="p-2 rounded-lg hover:bg-accent transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      title="Last page"
    >
      <ChevronsRight class="h-5 w-5" />
    </button>
  </div>
</template>

