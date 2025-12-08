import { createRouter, createWebHistory } from 'vue-router'
import Layout from '@/layouts/MainLayout.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      component: Layout,
      children: [
        {
          path: '',
          name: 'overview',
          component: () => import('@/views/OverviewView.vue'),
        },
        {
          path: 'browse',
          name: 'browse',
          component: () => import('@/views/BrowseView.vue'),
        },
        {
          path: 'issue/:issueId/pages',
          name: 'issue-gallery',
          component: () => import('@/views/IssueGalleryView.vue'),
        },
        {
          path: 'issue/:issueId',
          name: 'issue',
          component: () => import('@/views/IssueView.vue'),
        },
        {
          path: 'search',
          name: 'search',
          component: () => import('@/views/SearchView.vue'),
        },
        {
          path: 'entities',
          name: 'entities',
          component: () => import('@/views/EntitiesView.vue'),
        },
        {
          path: 'concepts',
          name: 'concepts',
          component: () => import('@/views/ConceptsView.vue'),
        },
        {
          path: 'keywords',
          name: 'keywords',
          component: () => import('@/views/KeywordsView.vue'),
        },
        {
          path: 'layout',
          name: 'layout',
          component: () => import('@/views/LayoutView.vue'),
        },
        {
          path: 'pictures',
          name: 'pictures',
          component: () => import('@/views/PicturesView.vue'),
        },
        {
          path: 'topics',
          name: 'topics',
          component: () => import('@/views/TopicsView.vue'),
        },
        {
          path: 'emotions',
          name: 'emotions',
          component: () => import('@/views/EmotionsView.vue'),
        },
        {
          path: 'preprocessing',
          name: 'preprocessing',
          component: () => import('@/views/PreprocessingView.vue'),
        },
      ],
    },
  ],
})

export default router
