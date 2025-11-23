<script setup lang="ts">
import { Search, X } from 'lucide-vue-next'

export interface SearchFilterOptions {
  query: string
}

interface Props {
  modelValue: SearchFilterOptions
  totalCount?: number
  loading?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  totalCount: 0,
  loading: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: SearchFilterOptions]
  'search': []
}>()

function updateFilter(key: keyof SearchFilterOptions, value: any) {
  emit('update:modelValue', {
    ...props.modelValue,
    [key]: value,
  })
}

function clearQuery() {
  updateFilter('query', '')
}
</script>

<template>
  <div class="flex flex-wrap items-center gap-4">
    <!-- Search Input -->
    <div class="relative flex-1 min-w-[300px]">
      <div class="relative">
        <Search class="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <input
          :value="modelValue.query"
          @input="updateFilter('query', ($event.target as HTMLInputElement).value)"
          @keydown.enter="$emit('search')"
          type="text"
          placeholder="Search text content..."
          class="flex h-9 w-full rounded-md border border-input bg-background pl-9 pr-8 py-1 text-sm shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
        />
        <button
          v-if="modelValue.query"
          @click="clearQuery"
          class="absolute right-2 top-2.5 text-muted-foreground hover:text-foreground"
        >
          <X class="h-4 w-4" />
        </button>
      </div>
    </div>

    <!-- Search Button -->
    <button
      @click="$emit('search')"
      :disabled="loading || !modelValue.query"
      class="inline-flex items-center justify-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow hover:bg-primary/90 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 h-9"
    >
      <Search class="h-4 w-4" />
      <span>Search</span>
    </button>

    <!-- Results Count -->
    <div v-if="totalCount > 0" class="text-sm text-muted-foreground">
      Found {{ totalCount.toLocaleString() }} results
    </div>
  </div>
</template>
