<script setup lang="ts">
import AnalysisHeader from '@/components/AnalysisHeader.vue'
</script>

<template>
  <div class="h-full flex flex-col overflow-auto">
    <!-- Header -->
    <div class="sticky top-0 z-10 bg-background px-4 pt-4 pb-6">
      <div class="flex flex-wrap items-start gap-4">
        <!-- Title -->
        <div class="flex items-center gap-2 min-w-0">
          <AnalysisHeader
            title="Concepts"
            description="Explore concept extraction and knowledge graphs"
            icon="concepts"
          />
        </div>
      </div>
    </div>

    <!-- Content area -->
    <div class="px-4 pb-6 space-y-6">
      <div class="rounded-lg border bg-card p-12 text-center">
        <p class="text-muted-foreground">Coming soon...</p>
      </div>
    </div>
  </div>
</template>
