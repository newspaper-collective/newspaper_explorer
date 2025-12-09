<script setup lang="ts">
import { ref, computed } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { useSourceStore } from '@/stores/source'
import {
  Home,
  Calendar,
  Search,
  Users,
  Network,
  Tag,
  LayoutDashboard,
  Image,
  MessageSquare,
  Smile,
  ChevronLeft,
  Globe,
  Building,
  FileText,
  Copyright,
  CalendarDays,
  Database,
  AlignLeft,
  Info,
  Wand2,
} from 'lucide-vue-next'
import {
  SliderRoot,
  SliderTrack,
  SliderRange,
  SliderThumb,
} from 'radix-vue'

type NavigationItem =
  | { name: string; to: string; icon: any; separator?: never }
  | { name: string; separator: true; to?: never; icon?: never }

const route = useRoute()
const sourceStore = useSourceStore()
const sidebarOpen = ref(true)

const navigation: NavigationItem[] = [
  { name: 'Overview', to: '/', icon: Home },
  { name: 'separator-1', separator: true },
  { name: 'Browse', to: '/browse', icon: Calendar },
  { name: 'Search', to: '/search', icon: Search },
  { name: 'Preprocessing', to: '/preprocessing', icon: Wand2 },
  { name: 'Data', to: '/data', icon: Database },
  { name: 'separator-2', separator: true },
  { name: 'Layout', to: '/layout', icon: LayoutDashboard },
  { name: 'Pictures', to: '/pictures', icon: Image },
  { name: 'separator-3', separator: true },
  { name: 'Concepts', to: '/concepts', icon: Network },
  { name: 'Emotions', to: '/emotions', icon: Smile },
  { name: 'Entities', to: '/entities', icon: Users },
  { name: 'Keywords', to: '/keywords', icon: Tag },
  { name: 'Topics', to: '/topics', icon: MessageSquare },
]

const toggleSidebar = () => {
  sidebarOpen.value = !sidebarOpen.value
}



// Helper to format numbers
const formatNumber = (num: number) => new Intl.NumberFormat().format(num)

// Icon map for analysis types
const analysisIconMap: Record<string, any> = {
  entities: Users,
  emotions: Smile,
  topics: MessageSquare,
  keywords: Tag,
  concepts: Network,
  layout: LayoutDashboard,
}

// Date Range Logic
const availableYears = computed(() => {
  if (!sourceStore.sourceStats?.years_available) return []
  return [...sourceStore.sourceStats.years_available].sort((a, b) => a - b)
})

const minYear = computed(() => availableYears.value[0] ?? 1900)
const maxYear = computed(() => availableYears.value[availableYears.value.length - 1] ?? new Date().getFullYear())

const selectedYear = computed({
  get: () => {
    if (!sourceStore.startDate || !sourceStore.endDate) return ''
    const startYear = new Date(sourceStore.startDate).getFullYear()
    const endYear = new Date(sourceStore.endDate).getFullYear()
    return startYear === endYear ? startYear.toString() : ''
  },
  set: (year: string) => {
    if (!year) return
    sourceStore.startDate = `${year}-01-01`
    sourceStore.endDate = `${year}-12-31`
  }
})

const yearRange = computed({
  get: () => {
    const start = sourceStore.startDate ? new Date(sourceStore.startDate).getFullYear() : minYear.value
    const end = sourceStore.endDate ? new Date(sourceStore.endDate).getFullYear() : maxYear.value
    return [start, end]
  },
  set: (range: number[]) => {
    if (!range || range.length !== 2) return
    sourceStore.startDate = `${range[0]}-01-01`
    sourceStore.endDate = `${range[1]}-12-31`
  }
})

</script>

<template>
  <div class="flex h-screen bg-background">
    <!-- Sidebar -->
    <aside
      :class="[
        'fixed inset-y-0 left-0 z-50 flex w-60 flex-col bg-card border-r transition-transform duration-200 overflow-y-auto',
        sidebarOpen ? 'translate-x-0' : '-translate-x-full',
      ]"
    >
      <!-- Header -->
      <div class="flex h-14 items-center px-4 border-b shrink-0">
        <span class="font-semibold text-lg">📰 Newspaper Explorer</span>
      </div>

      <div class="flex-1 p-4 space-y-3">
        <!-- Source Selector -->


        <!-- Current Source Info -->
        <div v-if="sourceStore.sourceInfo" class="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div class="flex flex-col space-y-1.5 p-4 pb-2">
            <h3 class="text-sm font-medium text-muted-foreground">Selected Source</h3>
          </div>
          <div class="p-4 pt-0 space-y-2">
            <div class="font-bold text-base">{{ sourceStore.sourceInfo.metadata.newspaper_title }}</div>

            <!-- Language and Years -->
            <div class="flex items-center gap-3 text-xs text-muted-foreground">
              <div class="flex items-center gap-1.5">
                <Globe class="h-3.5 w-3.5" />
                <span class="uppercase">{{ sourceStore.sourceInfo.metadata.language }}</span>
              </div>
              <span>•</span>
              <div class="flex items-center gap-1.5">
                <CalendarDays class="h-3.5 w-3.5" />
                <span>{{ sourceStore.sourceInfo.metadata.years_available }}</span>
              </div>
            </div>

            <!-- Provider -->
            <div v-if="sourceStore.sourceInfo.metadata.source_provider" class="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Building class="h-3.5 w-3.5 shrink-0" />
              <span class="truncate">{{ sourceStore.sourceInfo.metadata.source_provider }}</span>
            </div>

            <!-- Archive Size -->
            <div v-if="sourceStore.sourceInfo.total_archive_size" class="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Database class="h-3.5 w-3.5 shrink-0" />
              <span>{{ sourceStore.sourceInfo.total_archive_size }}</span>
            </div>

            <!-- Document Count -->
            <div v-if="sourceStore.sourceStats" class="flex items-center gap-1.5 text-xs text-muted-foreground">
              <FileText class="h-3.5 w-3.5 shrink-0" />
              <span>{{ formatNumber(sourceStore.sourceStats.total_issues) }} issues</span>
            </div>

            <!-- License -->
            <div v-if="sourceStore.sourceInfo.metadata.license" class="flex items-center gap-1.5 text-xs text-muted-foreground">
              <Copyright class="h-3.5 w-3.5 shrink-0" />
              <span class="truncate">{{ sourceStore.sourceInfo.metadata.license }}</span>
            </div>

            <!-- Collapsible More Info -->
            <details v-if="sourceStore.sourceInfo.metadata.description || sourceStore.sourceInfo.metadata.citation" class="group mt-2">
              <summary class="flex cursor-pointer items-center gap-1 text-xs font-medium text-muted-foreground hover:text-foreground">
                <Info class="h-3.5 w-3.5" />
                <span>More Info</span>
              </summary>
              <div class="mt-2 space-y-2 text-xs text-muted-foreground pl-5">
                <p v-if="sourceStore.sourceInfo.metadata.description" class="leading-relaxed break-words">
                  {{ sourceStore.sourceInfo.metadata.description }}
                </p>
                <div v-if="sourceStore.sourceInfo.metadata.citation" class="pt-1 border-t">
                  <div class="font-semibold mb-1">Citation:</div>
                  <div class="italic leading-relaxed break-words overflow-wrap-anywhere">{{ sourceStore.sourceInfo.metadata.citation }}</div>
                </div>
              </div>
            </details>
          </div>
        </div>
        <div v-else class="text-sm text-muted-foreground text-center py-4">
          No source selected
        </div>

        <!-- Date Range Filter -->
        <div class="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div class="flex flex-col space-y-1.5 p-4 pb-2">
            <h3 class="text-sm font-medium text-muted-foreground">Date Range Filter</h3>
          </div>
          <div class="p-4 pt-0 space-y-4">
            <!-- Year Selection -->
            <div class="space-y-2">
              <label class="text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70">Year</label>
              <select
                v-model="selectedYear"
                class="h-8 w-full px-3 py-2 text-xs"
              >
                <option value="">Select Year</option>
                <option v-for="year in availableYears" :key="year" :value="year.toString()">
                  {{ year }}
                </option>
              </select>
            </div>

            <!-- Range Slider -->
            <div class="space-y-4 pt-2">
              <SliderRoot
                v-model="yearRange"
                :min="minYear"
                :max="maxYear"
                :step="1"
                class="relative flex w-full touch-none select-none items-center"
              >
                <SliderTrack class="relative h-1.5 w-full grow overflow-hidden rounded-full bg-secondary">
                  <SliderRange class="absolute h-full bg-primary" />
                </SliderTrack>
                <SliderThumb
                  class="block h-4 w-4 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                />
                <SliderThumb
                  class="block h-4 w-4 rounded-full border-2 border-primary bg-background ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50"
                />
              </SliderRoot>
              <div class="flex justify-between text-xs text-muted-foreground">
                <span>{{ minYear }}</span>
                <span>{{ maxYear }}</span>
              </div>
            </div>

            <div class="space-y-2">
              <div class="flex items-center gap-2">
                <label for="start-date" class="text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 w-12 flex-shrink-0">From</label>
                <input
                  id="start-date"
                  type="date"
                  v-model="sourceStore.startDate"
                  class="flex h-8 flex-1 rounded-md border border-input bg-background px-3 py-1 text-xs shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
              <div class="flex items-center gap-2">
                <label for="end-date" class="text-xs font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70 w-12 flex-shrink-0">To</label>
                <input
                  id="end-date"
                  type="date"
                  v-model="sourceStore.endDate"
                  class="flex h-8 flex-1 rounded-md border border-input bg-background px-3 py-1 text-xs shadow-sm transition-colors file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Available Data -->
        <div v-if="sourceStore.sourceInfo" class="rounded-lg border bg-card text-card-foreground shadow-sm">
          <div class="flex flex-col space-y-1.5 p-4 pb-2">
            <h3 class="text-sm font-medium text-muted-foreground">Available Data</h3>
          </div>
          <div class="p-4 pt-0 space-y-3">
            <!-- Loading state -->
            <div v-if="sourceStore.loading" class="text-xs text-muted-foreground italic">
              Loading data information...
            </div>

            <!-- Analysis Results -->
            <div v-else-if="sourceStore.sourceInfo.analysis_results && Object.keys(sourceStore.sourceInfo.analysis_results).length > 0">
              <div class="text-xs font-semibold mb-2 text-muted-foreground">Analysis Results:</div>
              <div class="space-y-1.5">
                <div
                  v-for="[type, summary] in Object.entries(sourceStore.sourceInfo.analysis_results)"
                  :key="type"
                  class="flex items-center gap-1.5 text-xs text-muted-foreground"
                >
                  <component :is="analysisIconMap[type] || FileText" class="h-3.5 w-3.5 shrink-0" />
                  <span class="capitalize">{{ type }}:</span>
                  <span>{{ summary.count }} {{ summary.count === 1 ? 'file' : 'files' }}</span>
                </div>
              </div>
            </div>

            <!-- Preprocessed Data -->
            <div v-if="!sourceStore.loading && sourceStore.sourceStats && (sourceStore.sourceStats.total_lines > 0 || sourceStore.sourceStats.total_blocks > 0)">
              <div v-if="sourceStore.sourceInfo.analysis_results && Object.keys(sourceStore.sourceInfo.analysis_results).length > 0" class="h-px bg-border my-2" />
              <div class="text-xs font-semibold mb-2 text-muted-foreground">Preprocessed Data:</div>
              <div class="space-y-1.5">
                <div v-if="sourceStore.sourceStats.total_lines > 0" class="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <FileText class="h-3.5 w-3.5 shrink-0" />
                  <span>Lines: {{ formatNumber(sourceStore.sourceStats.total_lines) }}</span>
                </div>
                <div v-if="sourceStore.sourceStats.total_blocks > 0" class="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <AlignLeft class="h-3.5 w-3.5 shrink-0" />
                  <span>Text Blocks: {{ formatNumber(sourceStore.sourceStats.total_blocks) }}</span>
                </div>
              </div>
            </div>

            <!-- No data message -->
            <div v-if="!sourceStore.loading && (!sourceStore.sourceInfo.analysis_results || Object.keys(sourceStore.sourceInfo.analysis_results).length === 0) && (!sourceStore.sourceStats || (sourceStore.sourceStats.total_lines === 0 && sourceStore.sourceStats.total_blocks === 0))" class="text-xs text-muted-foreground">
              <p class="mb-2">No data available yet.</p>
              <p class="text-xs">Parse the source with:</p>
              <code class="block mt-1 px-2 py-1 bg-muted rounded text-xs">
                newspaper-explorer data parse --source {{ sourceStore.currentSource }}
              </code>
            </div>
          </div>
        </div>
      </div>
    </aside>

    <!-- Main content -->
    <div
      :class="[
        'flex flex-1 flex-col transition-all duration-200',
        sidebarOpen ? 'ml-60' : 'ml-0',
      ]"
    >
      <!-- Top bar -->
      <header class="sticky top-0 z-40 flex h-14 items-center gap-4 border-b bg-card px-4">
        <button
          @click="toggleSidebar"
          class="rounded-md p-2 hover:bg-accent"
        >
          <ChevronLeft
            :class="[
              'h-5 w-5 transition-transform duration-200',
              !sidebarOpen && 'rotate-180'
            ]"
          />
        </button>

        <h2 class="text-lg font-semibold">
          {{ sourceStore.sourceInfo?.metadata?.newspaper_title || 'Historical Newspapers' }}
        </h2>

        <!-- Navigation Tabs -->
        <nav class="ml-auto flex items-center gap-1">
          <template v-for="item in navigation" :key="item.name">
            <!-- Separator -->
            <div
              v-if="item.separator"
              class="h-6 w-px bg-border mx-1"
            />
            <!-- Navigation Link -->
            <RouterLink
              v-else
              :to="item.to"
              :class="[
                route.path === item.to
                  ? 'bg-accent text-accent-foreground'
                  : 'text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground',
                'flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors',
              ]"
            >
              <component :is="item.icon" class="h-4 w-4" />
              <span class="hidden lg:inline">{{ item.name }}</span>
            </RouterLink>
          </template>
        </nav>
      </header>

      <!-- Page content -->
      <main class="flex-1 overflow-y-auto">
        <RouterView />
      </main>
    </div>
  </div>
</template>
