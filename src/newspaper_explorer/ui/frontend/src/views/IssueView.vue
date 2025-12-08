<script setup lang="ts">
import { onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import IssueReader from '@/components/IssueReader.vue'

const route = useRoute()
const router = useRouter()

const issueId = route.params.issueId as string

function goBack() {
  // Go back to gallery view
  router.push({ name: 'issue-gallery', params: { issueId } })
}

onMounted(() => {
  // If no page parameter is provided, redirect to gallery
  if (!route.query.page) {
    router.replace({ name: 'issue-gallery', params: { issueId } })
  }
})
</script>

<template>
  <IssueReader
    :issue-id="issueId"
    :show-back-button="true"
    @back="goBack"
  />
</template>
