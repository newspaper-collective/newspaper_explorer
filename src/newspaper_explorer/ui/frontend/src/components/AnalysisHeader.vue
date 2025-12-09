<script setup lang="ts">
import { computed } from 'vue'
import {
  Users,
  Smile,
  BookOpen,
  Tag,
  Network,
  LayoutGrid,
  Search,
  Wand2,
  Database,
  Image,
} from 'lucide-vue-next'

interface Props {
  title: string
  description: string
  icon?: 'entities' | 'emotions' | 'topics' | 'keywords' | 'concepts' | 'layout' | 'search' | 'preprocessing' | 'data' | 'pictures'
}

const props = defineProps<Props>()

const iconComponents = {
  entities: Users,
  emotions: Smile,
  topics: BookOpen,
  keywords: Tag,
  concepts: Network,
  layout: LayoutGrid,
  search: Search,
  preprocessing: Wand2,
  data: Database,
  pictures: Image,
}

const IconComponent = computed(() => props.icon ? iconComponents[props.icon] : null)
</script>

<template>
  <div class="flex items-start gap-2 h-full w-fit">
    <div v-if="icon && IconComponent" class="flex-shrink-0 flex items-center h-7">
      <component :is="IconComponent" :size="20" />
    </div>
    <div>
      <h1 class="text-lg font-medium">{{ title }}</h1>
      <p v-if="description" class="text-sm text-muted-foreground mt-1">
        {{ description }}
      </p>
    </div>
  </div>
</template>
