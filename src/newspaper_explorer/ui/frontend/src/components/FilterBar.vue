<script setup lang="ts">
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
      <input
        :value="modelValue.searchQuery"
        @input="updateFilter('searchQuery', ($event.target as HTMLInputElement).value)"
        type="text"
        placeholder="Search by page ID, caption text..."
        class="w-full rounded-md border border-input bg-background px-2 py-1 text-sm"
      />

      <!-- Filter checkboxes -->
      <div class="flex flex-wrap gap-4">
        <label
          v-if="showCaptionFilter"
          class="flex items-center gap-2 text-xs cursor-pointer"
        >
          <input
            :checked="modelValue.onlyWithCaptions"
            @change="updateFilter('onlyWithCaptions', ($event.target as HTMLInputElement).checked)"
            type="checkbox"
            class="rounded border-input"
          />
          <span class="text-muted-foreground">Only with captions</span>
        </label>

        <label
          v-if="showHeaderFooterFilter"
          class="flex items-center gap-2 text-xs cursor-pointer"
        >
          <input
            :checked="modelValue.excludeHeadersFooters"
            @change="updateFilter('excludeHeadersFooters', ($event.target as HTMLInputElement).checked)"
            type="checkbox"
            class="rounded border-input"
          />
          <span class="text-muted-foreground">Exclude headers/footers</span>
        </label>
      </div>

      <!-- Height Filter and Header/Footer Threshold Row -->
      <div class="flex items-center justify-between gap-3">
        <!-- Min Height Filter -->
        <div v-if="showHeightFilter" class="flex items-center gap-2 flex-1">
          <label class="text-xs text-muted-foreground whitespace-nowrap">
            Min Height: {{ modelValue.minHeight }}px
          </label>
          <input
            :value="modelValue.minHeight"
            @input="updateFilter('minHeight', parseInt(($event.target as HTMLInputElement).value))"
            type="range"
            min="0"
            max="1000"
            step="50"
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
          <input
            :value="modelValue.headerFooterThreshold"
            @input="updateFilter('headerFooterThreshold', parseInt(($event.target as HTMLInputElement).value))"
            type="range"
            min="0"
            max="30"
            step="1"
            class="flex-1"
          />
        </div>
      </div>

      <!-- Confidence Filter Row with Stats -->
      <div class="flex items-center gap-2">
        <label class="text-xs text-muted-foreground whitespace-nowrap">
          Conf: {{ modelValue.minConfidence }}%
        </label>
        <input
          :value="modelValue.minConfidence"
          @input="updateFilter('minConfidence', parseInt(($event.target as HTMLInputElement).value))"
          type="range"
          min="0"
          max="100"
          step="5"
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
