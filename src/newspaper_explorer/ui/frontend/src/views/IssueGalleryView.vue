<script setup lang="ts">
import { useRoute, useRouter } from 'vue-router'
import PageThumbnailGallery from '@/components/PageThumbnailGallery.vue'

const route = useRoute()
const router = useRouter()

const issueId = route.params.issueId as string

function goBack() {
  router.push({ name: 'browse' })
}
</script>

<template>
  <PageThumbnailGallery
    :issue-id="issueId"
    :show-back-button="true"
    @back="goBack"
  />
</template>
