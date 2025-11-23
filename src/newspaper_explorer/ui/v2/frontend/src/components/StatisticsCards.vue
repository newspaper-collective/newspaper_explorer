<script setup lang="ts">
import { computed } from 'vue'

export interface StatCard {
  label: string
  value: number | string
  format?: 'number' | 'percentage' | 'pixels' | 'custom'
  decimals?: number
}

interface Props {
  stats: StatCard[]
  columns?: number
}

const props = withDefaults(defineProps<Props>(), {
  columns: 5,
})

function formatValue(stat: StatCard): string {
  if (typeof stat.value === 'string') {
    return stat.value
  }

  const decimals = stat.decimals ?? (stat.format === 'percentage' ? 1 : 0)

  switch (stat.format) {
    case 'percentage':
      return `${(stat.value * 100).toFixed(decimals)}%`
    case 'pixels':
      return `${stat.value.toFixed(decimals)}px`
    case 'number':
    default:
      return stat.value.toLocaleString()
  }
}

const gridClass = computed(() => {
  const colMap: Record<number, string> = {
    2: 'md:grid-cols-2',
    3: 'md:grid-cols-3',
    4: 'md:grid-cols-4',
    5: 'md:grid-cols-5',
    6: 'md:grid-cols-6',
  }
  return colMap[props.columns] || 'md:grid-cols-5'
})
</script>

<template>
  <div class="grid gap-4" :class="gridClass">
    <div
      v-for="stat in stats"
      :key="stat.label"
      class="rounded-lg border bg-card p-4"
    >
      <div class="text-sm text-muted-foreground">{{ stat.label }}</div>
      <div class="text-2xl font-bold">{{ formatValue(stat) }}</div>
    </div>
  </div>
</template>
