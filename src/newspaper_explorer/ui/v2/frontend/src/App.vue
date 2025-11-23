<script setup lang="ts">
import { RouterView } from 'vue-router'
import { useSourceStore } from '@/stores/source'
import { onMounted } from 'vue'

const sourceStore = useSourceStore()

onMounted(async () => {
  await sourceStore.loadSources()
})
</script>

<template>
  <div class="min-h-screen bg-background">
    <RouterView />
  </div>
</template>
