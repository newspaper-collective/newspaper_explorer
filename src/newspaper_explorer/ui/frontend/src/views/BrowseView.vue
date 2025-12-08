<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useSourceStore } from '@/stores/source'
import { useRouter, useRoute } from 'vue-router'
import api from '@/lib/api.ts'
import { Home } from 'lucide-vue-next'
import PaginationControls from '@/components/PaginationControls.vue'

const sourceStore = useSourceStore()
const router = useRouter()
const route = useRoute()

// Navigation state
type ViewMode = 'year' | 'month' | 'issue'
const viewMode = ref<ViewMode>('year')
const selectedYear = ref<number | null>(null)
const selectedMonth = ref<number | null>(null)

// Filter state
const yearFrom = ref<number | null>(null)
const yearTo = ref<number | null>(null)
const sortOrder = ref<'asc' | 'desc'>('asc')

// Data state
const years = ref<any[]>([])
const months = ref<any[]>([])
const issues = ref<any[]>([])
const loading = ref(false)

// Pagination state
const currentPage = ref(1)
const pageSize = ref(20)
const totalPages = ref(1)
const totalCount = ref(0)

const monthNames = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

const monthNamesShort = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
]

// Computed properties
const sourceTitle = computed(() => {
  return sourceStore.sourceInfo?.metadata?.newspaper_title || 'Collection'
})

const breadcrumbs = computed(() => {
  const crumbs = [{ label: sourceTitle.value, mode: 'year' as ViewMode }]

  if (selectedYear.value !== null) {
    crumbs.push({ label: `${selectedYear.value}`, mode: 'month' as ViewMode })
  }

  if (selectedMonth.value !== null) {
    crumbs.push({
      label: monthNames[selectedMonth.value - 1],
      mode: 'issue' as ViewMode
    })
  }

  return crumbs
})

const statsText = computed(() => {
  if (viewMode.value === 'year') {
    const totalLines = years.value.reduce((sum, y) => sum + (y.line_count || 0), 0)
    return {
      primary: `${years.value.length} years`,
      secondary: `${totalLines.toLocaleString()} total lines`
    }
  } else if (viewMode.value === 'month') {
    return {
      primary: `Page ${currentPage.value} of ${totalPages.value}`,
      secondary: `${months.value.length} months on this page`
    }
  } else {
    return {
      primary: `Page ${currentPage.value} of ${totalPages.value}`,
      secondary: `${totalCount.value} total issues`
    }
  }
})

// Navigation functions
function updateUrl() {
  const query: any = {}

  if (selectedYear.value !== null) {
    query.year = selectedYear.value.toString()
  }
  if (selectedMonth.value !== null) {
    query.month = selectedMonth.value.toString()
  }
  if (currentPage.value > 1) {
    query.page = currentPage.value.toString()
  }
  if (yearFrom.value !== null) {
    query.yearFrom = yearFrom.value.toString()
  }
  if (yearTo.value !== null) {
    query.yearTo = yearTo.value.toString()
  }
  if (sortOrder.value !== 'asc') {
    query.sort = sortOrder.value
  }

  router.replace({ query })
}

function navigateToYear(year: number | null) {
  selectedYear.value = year
  selectedMonth.value = null
  currentPage.value = 1

  if (year === null) {
    viewMode.value = 'year'
    loadYears()
  } else {
    viewMode.value = 'month'
    loadMonths()
  }
  updateUrl()
}

function navigateToMonth(year: number, month: number | null) {
  selectedYear.value = year
  selectedMonth.value = month
  currentPage.value = 1

  if (month === null) {
    viewMode.value = 'month'
    loadMonths()
  } else {
    viewMode.value = 'issue'
    loadIssues()
  }
  updateUrl()
}

function openIssue(issueId: string) {
  router.push({ name: 'issue-gallery', params: { issueId } })
}

function resetFilters() {
  yearFrom.value = null
  yearTo.value = null
  sortOrder.value = 'asc'

  if (selectedYear.value !== null) {
    navigateToYear(null)
  } else {
    loadYears()
  }
}

// Data loading functions
async function loadYears() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    const params: any = { sort_order: sortOrder.value }
    if (yearFrom.value) params.year_from = yearFrom.value
    if (yearTo.value) params.year_to = yearTo.value

    const response = await api.get(
      `/data/${sourceStore.currentSource}/browse/years`,
      { params }
    )
    years.value = response.data
  } catch (error) {
    console.error('Failed to load years:', error)
  } finally {
    loading.value = false
  }
}

async function loadMonths() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    const params: any = {
      sort_order: sortOrder.value,
      page: currentPage.value,
      page_size: pageSize.value
    }

    if (selectedYear.value !== null) {
      params.year = selectedYear.value
    } else {
      if (yearFrom.value) params.year_from = yearFrom.value
      if (yearTo.value) params.year_to = yearTo.value
    }

    const response = await api.get(
      `/data/${sourceStore.currentSource}/browse/months`,
      { params }
    )

    months.value = response.data.results
    totalPages.value = response.data.total_pages
    totalCount.value = response.data.total
  } catch (error) {
    console.error('Failed to load months:', error)
  } finally {
    loading.value = false
  }
}

async function loadIssues() {
  if (!sourceStore.currentSource) return

  loading.value = true
  try {
    const params: any = {
      sort_order: sortOrder.value,
      page: currentPage.value,
      page_size: pageSize.value
    }

    if (selectedYear.value !== null) {
      params.year = selectedYear.value
    }
    if (selectedMonth.value !== null) {
      params.month = selectedMonth.value
    }

    const response = await api.get(
      `/data/${sourceStore.currentSource}/browse/issues`,
      { params }
    )

    issues.value = response.data.results
    totalPages.value = response.data.total_pages
    totalCount.value = response.data.total
  } catch (error) {
    console.error('Failed to load issues:', error)
  } finally {
    loading.value = false
  }
}

function applyFilters() {
  currentPage.value = 1
  if (viewMode.value === 'year') {
    loadYears()
  } else if (viewMode.value === 'month') {
    loadMonths()
  } else {
    loadIssues()
  }
  updateUrl()
}

// Watch for pagination changes
watch(currentPage, () => {
  if (viewMode.value === 'month') {
    loadMonths()
  } else if (viewMode.value === 'issue') {
    loadIssues()
  }
  updateUrl()
})

function formatDate(dateString: string): { weekday: string; date: string } {
  const date = new Date(dateString)
  const weekday = date.toLocaleDateString('en-US', { weekday: 'long' })
  const formatted = date.toLocaleDateString('de-DE')
  return { weekday, date: formatted }
}

function getImageUrl(imagePath: string | null): string | undefined {
  if (!imagePath || !sourceStore.currentSource) return undefined
  return `/static/${sourceStore.currentSource}/images/${imagePath}`
}

// Watch for source changes
watch(() => sourceStore.currentSource, () => {
  selectedYear.value = null
  selectedMonth.value = null
  viewMode.value = 'year'
  currentPage.value = 1
  loadYears()
})

// Watch for sort order changes when drilling down
watch(sortOrder, () => {
  if (selectedYear.value !== null) {
    currentPage.value = 1
    if (viewMode.value === 'month') {
      loadMonths()
    } else if (viewMode.value === 'issue') {
      loadIssues()
    }
  }
})

onMounted(() => {
  if (sourceStore.currentSource) {
    // Initialize from URL parameters
    const year = route.query.year ? parseInt(route.query.year as string, 10) : null
    const month = route.query.month ? parseInt(route.query.month as string, 10) : null
    const page = route.query.page ? parseInt(route.query.page as string, 10) : 1
    const yFrom = route.query.yearFrom ? parseInt(route.query.yearFrom as string, 10) : null
    const yTo = route.query.yearTo ? parseInt(route.query.yearTo as string, 10) : null
    const sort = route.query.sort as 'asc' | 'desc' || 'asc'

    // Set state from URL
    yearFrom.value = yFrom
    yearTo.value = yTo
    sortOrder.value = sort
    currentPage.value = page

    if (year !== null) {
      selectedYear.value = year
      if (month !== null) {
        selectedMonth.value = month
        viewMode.value = 'issue'
        loadIssues()
      } else {
        viewMode.value = 'month'
        loadMonths()
      }
    } else {
      loadYears()
    }
  }
})
</script>

<template>
  <div class="space-y-6 px-4 pb-4">
    <!-- Breadcrumb Navigation -->
    <div class="mt-4">
      <div class="flex items-start gap-2">
        <Home class="h-5 w-5 mt-1 flex-shrink-0" />
        <div>
          <div class="flex items-center gap-2 flex-wrap">
            <button
              v-for="(crumb, index) in breadcrumbs"
              :key="index"
              @click="() => {
                if (index === 0) navigateToYear(null)
                else if (index === 1 && selectedYear !== null) navigateToMonth(selectedYear, null)
              }"
              :class="[
                'flex items-center gap-2 text-lg font-medium transition-colors',
                index === breadcrumbs.length - 1
                  ? 'text-foreground cursor-default'
                  : 'text-muted-foreground hover:text-foreground cursor-pointer'
              ]"
            >
              {{ crumb.label }}
              <span v-if="index < breadcrumbs.length - 1" class="text-muted-foreground">›</span>
            </button>
          </div>
          <p class="text-sm text-muted-foreground mt-1">
            Navigate through years, months, and individual newspaper issues
          </p>
        </div>
      </div>
    </div>

    <!-- No source selected -->
    <div v-if="!sourceStore.currentSource" class="rounded-lg border bg-card p-8 text-center">
      <p class="text-muted-foreground">
        Please select a source from the sidebar to browse newspapers
      </p>
    </div>

    <!-- Main layout with sidebar and content -->
    <div v-else class="grid grid-cols-1 lg:grid-cols-[250px_1fr] gap-6">
      <!-- Left Sidebar - Filters -->
      <div class="space-y-4">
        <div class="rounded-lg border bg-card p-4 space-y-4">
          <h3 class="font-semibold">Browse Filters</h3>

          <!-- Year Filter -->
          <div class="space-y-2">
            <label class="text-sm font-medium">Time Period</label>

            <!-- Year range (when at year level) -->
            <div v-if="selectedYear === null" class="grid grid-cols-2 gap-2">
              <div>
                <label class="text-xs text-muted-foreground">From</label>
                <input
                  v-model.number="yearFrom"
                  type="number"
                  class="w-full px-3 py-2 text-sm rounded-md border bg-background"
                  placeholder="Year"
                />
              </div>
              <div>
                <label class="text-xs text-muted-foreground">To</label>
                <input
                  v-model.number="yearTo"
                  type="number"
                  class="w-full px-3 py-2 text-sm rounded-md border bg-background"
                  placeholder="Year"
                />
              </div>
            </div>

            <!-- Year dropdown (when drilling down) -->
            <select
              v-else
              v-model.number="selectedYear"
              @change="navigateToMonth(selectedYear!, null)"
              class="w-full px-3 py-2 text-sm rounded-md border bg-background"
            >
              <option v-for="year in years" :key="year.year" :value="year.year">
                {{ year.year }}
              </option>
            </select>
          </div>

          <!-- Month Filter (when year is selected) -->
          <div v-if="selectedYear !== null" class="space-y-2">
            <label class="text-sm font-medium">Month</label>
            <select
              v-model.number="selectedMonth"
              @change="selectedMonth !== null ? navigateToMonth(selectedYear!, selectedMonth!) : navigateToMonth(selectedYear!, null)"
              class="w-full px-3 py-2 text-sm rounded-md border bg-background"
            >
              <option :value="null">All Months</option>
              <option v-for="month in months" :key="month.month" :value="month.month">
                {{ monthNames[month.month - 1] }} ({{ month.issue_count }} issues)
              </option>
            </select>
          </div>

          <!-- Sort Order -->
          <div class="space-y-2">
            <label class="text-sm font-medium">Sort Order</label>
            <select
              v-model="sortOrder"
              class="w-full px-3 py-2 text-sm rounded-md border bg-background"
            >
              <option value="asc">Oldest First</option>
              <option value="desc">Newest First</option>
            </select>
          </div>

          <!-- Apply Button (only at year level) -->
          <button
            v-if="selectedYear === null"
            @click="applyFilters"
            class="w-full px-4 py-2 bg-primary text-primary-foreground rounded-md hover:bg-primary/90 transition-colors"
          >
            Apply Filters
          </button>

          <!-- Reset Button -->
          <button
            @click="resetFilters"
            class="w-full px-4 py-2 border rounded-md hover:bg-accent transition-colors text-sm"
          >
            Reset Filters
          </button>
        </div>

        <!-- Statistics -->
        <div class="rounded-lg border bg-card p-4 space-y-2">
          <h3 class="text-sm font-medium">Statistics</h3>
          <p class="text-sm">{{ statsText.primary }}</p>
          <p class="text-xs text-muted-foreground">{{ statsText.secondary }}</p>
        </div>
      </div>

      <!-- Right Content Area -->
      <div class="space-y-4">
        <!-- Loading State -->
        <div v-if="loading" class="text-center py-12">
          <p class="text-muted-foreground">Loading...</p>
        </div>

        <!-- Year View -->
        <div v-else-if="viewMode === 'year'" class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div
            v-for="year in years"
            :key="year.year"
            @click="navigateToMonth(year.year, null)"
            class="rounded-lg border bg-card overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
          >
            <div class="grid grid-cols-2 h-[200px]">
              <!-- Left - Year Info -->
              <div class="flex flex-col items-center justify-center p-4 text-center">
                <div class="text-4xl font-semibold">{{ year.year }}</div>
                <div class="w-3/5 h-px bg-border my-2"></div>
                <div class="space-y-1">
                  <p class="text-sm font-medium">{{ year.issue_count }} issues</p>
                  <p class="text-xs text-muted-foreground">
                    {{ year.line_count.toLocaleString() }} lines
                  </p>
                </div>
              </div>
              <!-- Right - Image -->
              <div class="bg-muted relative overflow-hidden">
                <img
                  v-if="year.image_path"
                  :src="getImageUrl(year.image_path)"
                  class="w-full h-full object-cover"
                  alt="Year preview"
                />
              </div>
            </div>
          </div>
        </div>

        <!-- Month View -->
        <div v-else-if="viewMode === 'month'" class="space-y-4">
          <div class="grid gap-4 grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            <div
              v-for="month in months"
              :key="`${month.year}-${month.month}`"
              @click="navigateToMonth(month.year, month.month)"
              class="rounded-lg border bg-card p-4 hover:shadow-lg transition-shadow cursor-pointer"
            >
              <div class="flex flex-col items-center text-center space-y-2">
                <div class="text-2xl font-semibold">
                  {{ monthNamesShort[month.month - 1] }}
                </div>
                <div v-if="selectedYear === null" class="text-sm text-muted-foreground font-medium">
                  {{ month.year }}
                </div>
                <div class="w-3/5 h-px bg-border"></div>
                <div class="space-y-1">
                  <p class="text-sm font-medium">{{ month.issue_count }} issues</p>
                  <p class="text-xs text-muted-foreground">
                    {{ month.line_count.toLocaleString() }} lines
                  </p>
                </div>
              </div>
            </div>
          </div>

          <!-- Pagination -->
          <PaginationControls
            :current-page="currentPage"
            :total-pages="totalPages"
            :loading="loading"
            @update:current-page="(page) => currentPage = page"
          />
        </div>

        <!-- Issue View -->
        <div v-else-if="viewMode === 'issue'" class="space-y-4">
          <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            <div
              v-for="issue in issues"
              :key="issue.issue_id"
              @click="openIssue(issue.issue_id)"
              class="rounded-lg border bg-card overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
            >
              <div class="grid grid-cols-2 h-[200px]">
                <!-- Left - Issue Info -->
                <div class="flex flex-col justify-center p-4">
                  <div class="space-y-1">
                    <p class="text-xs text-muted-foreground uppercase">
                      {{ formatDate(issue.date).weekday }}
                      <span v-if="issue.daily_count"> #{{ issue.daily_count }}</span>
                    </p>
                    <p class="text-lg font-semibold">{{ formatDate(issue.date).date }}</p>
                  </div>
                  <div class="w-4/5 h-px bg-border my-2"></div>
                  <div class="space-y-1">
                    <p class="text-sm text-muted-foreground">
                      Issue: {{ issue.issue_number || 'N/A' }}
                    </p>
                    <div class="flex gap-2 flex-wrap mt-2">
                      <span class="text-xs px-2 py-1 rounded-full border">
                        {{ issue.page_count }} pages
                      </span>
                      <span class="text-xs px-2 py-1 rounded-full border">
                        {{ issue.block_count }} blocks
                      </span>
                    </div>
                  </div>
                </div>
                <!-- Right - Image -->
                <div class="bg-muted relative overflow-hidden">
                  <img
                    v-if="issue.image_path"
                    :src="getImageUrl(issue.image_path)"
                    class="w-full h-full object-cover"
                    alt="Issue preview"
                  />
                </div>
              </div>
            </div>
          </div>

          <!-- Pagination -->
          <PaginationControls
            :current-page="currentPage"
            :total-pages="totalPages"
            :loading="loading"
            @update:current-page="(page) => currentPage = page"
          />
        </div>
      </div>
    </div>
  </div>
</template>
