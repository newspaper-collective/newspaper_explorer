<script setup lang="ts">
import { Search, X } from 'lucide-vue-next'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'

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
        <Input
          :model-value="modelValue.query"
          @update:model-value="updateFilter('query', $event)"
          @keydown.enter="$emit('search')"
          type="text"
          placeholder="Search text content..."
          class="pl-9 pr-8"
        />
        <Button
          v-if="modelValue.query"
          @click="clearQuery"
          variant="ghost"
          size="icon"
          class="absolute right-0 top-0 h-9 w-9 text-muted-foreground hover:text-foreground"
        >
          <X class="h-4 w-4" />
        </Button>
      </div>
    </div>

    <!-- Search Button -->
    <Button
      @click="$emit('search')"
      :disabled="loading || !modelValue.query"
    >
      <Search class="h-4 w-4" />
      <span>Search</span>
    </Button>

    <!-- Results Count -->
    <div v-if="totalCount > 0" class="text-sm text-muted-foreground">
      Found {{ totalCount.toLocaleString() }} results
    </div>
  </div>
</template>
