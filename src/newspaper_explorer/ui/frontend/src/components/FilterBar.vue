<script setup lang="ts">
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { Slider } from '@/components/ui/slider'

export interface FilterOptions {
  searchQuery: string
  minConfidence: number
  onlyWithCaptions: boolean
  excludeHeadersFooters: boolean
  headerFooterThreshold: number
  minHeight: number
}

interface Props {
  modelValue: FilterOptions
  filteredCount: number
  totalCount: number
  showCaptionFilter?: boolean
  showHeaderFooterFilter?: boolean
  showHeightFilter?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  showCaptionFilter: true,
  showHeaderFooterFilter: true,
  showHeightFilter: true,
})

const emit = defineEmits<{
  'update:modelValue': [value: FilterOptions]
}>()

function updateFilter<K extends keyof FilterOptions>(key: K, value: FilterOptions[K]) {
  emit('update:modelValue', {
    ...props.modelValue,
    [key]: value,
  })
}
</script>

<template>
  <div class="rounded-lg border bg-card p-3 flex items-center self-stretch">
    <div class="flex flex-col gap-2 w-full">
      <!-- Search -->
      <Input
        :model-value="modelValue.searchQuery"
        @update:model-value="updateFilter('searchQuery', $event)"
        type="text"
        placeholder="Search by page ID, caption text..."
        class="h-8 text-sm"
      />

      <!-- Filter checkboxes -->
      <div class="flex flex-wrap gap-4">
        <div
          v-if="showCaptionFilter"
          class="flex items-center gap-2 text-xs"
        >
          <Checkbox
            :checked="modelValue.onlyWithCaptions"
            @update:checked="updateFilter('onlyWithCaptions', $event)"
            id="captions-filter"
          />
          <label for="captions-filter" class="text-muted-foreground cursor-pointer">Only with captions</label>
        </div>

        <div
          v-if="showHeaderFooterFilter"
          class="flex items-center gap-2 text-xs"
        >
          <Checkbox
            :checked="modelValue.excludeHeadersFooters"
            @update:checked="updateFilter('excludeHeadersFooters', $event)"
            id="header-footer-filter"
          />
          <label for="header-footer-filter" class="text-muted-foreground cursor-pointer">Exclude headers/footers</label>
        </div>
      </div>

      <!-- Height Filter and Header/Footer Threshold Row -->
      <div class="flex items-center justify-between gap-3">
        <!-- Min Height Filter -->
        <div v-if="showHeightFilter" class="flex items-center gap-2 flex-1">
          <label class="text-xs text-muted-foreground whitespace-nowrap">
            Min Height: {{ modelValue.minHeight }}px
          </label>
          <Slider
            :model-value="[modelValue.minHeight]"
            @update:model-value="updateFilter('minHeight', $event[0])"
            :min="0"
            :max="1000"
            :step="50"
            class="flex-1"
          />
        </div>

        <!-- Header/Footer Threshold (only when enabled) -->
        <div
          v-if="showHeaderFooterFilter && modelValue.excludeHeadersFooters"
          class="flex items-center gap-2 flex-1"
        >
          <label class="text-xs text-muted-foreground whitespace-nowrap">
            H/F: {{ modelValue.headerFooterThreshold }}%
          </label>
          <Slider
            :model-value="[modelValue.headerFooterThreshold]"
            @update:model-value="updateFilter('headerFooterThreshold', $event[0])"
            :min="0"
            :max="30"
            :step="1"
            class="flex-1"
          />
        </div>
      </div>

      <!-- Confidence Filter Row with Stats -->
      <div class="flex items-center gap-2">
        <label class="text-xs text-muted-foreground whitespace-nowrap">
          Conf: {{ modelValue.minConfidence }}%
        </label>
        <Slider
          :model-value="[modelValue.minConfidence]"
          @update:model-value="updateFilter('minConfidence', $event[0])"
          :min="0"
          :max="100"
          :step="5"
          class="flex-1"
        />

        <!-- Results count -->
        <span class="text-xs text-muted-foreground whitespace-nowrap">
          {{ filteredCount.toLocaleString() }}/{{ totalCount.toLocaleString() }}
        </span>
      </div>
    </div>
  </div>
</template>
