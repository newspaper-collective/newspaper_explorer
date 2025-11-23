<script setup lang="ts">
import { ref, watch, computed } from 'vue'
import { X, Tag, ChevronDown, ChevronRight, Frown, HeartHandshake, Smile, Skull, Flame, Zap, Users, MessageSquare, Network, LayoutDashboard, User, Building2, MapPin, Calendar } from 'lucide-vue-next'
import api from '@/lib/api'

interface Props {
  isOpen: boolean
  pageId: string | null
  sourceName: string
  sidebarMode?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  sidebarMode: false,
})
const emit = defineEmits<{
  close: []
}>()

const loading = ref(false)
const analysisData = ref<any>(null)

// Collapsible section states
const keywordsExpanded = ref(false)
const emotionsExpanded = ref(false)
const layoutExpanded = ref(false)
const entitiesExpanded = ref(false)

// Nested collapsible for details
const emotionDetailsExpanded = ref(false)
const layoutDetailsExpanded = ref(false)

// Selected result set for each type
const selectedKeywords = ref<string | null>(null)
const selectedEmotions = ref<string | null>(null)
const selectedLayout = ref<string | null>(null)
const selectedEntities = ref<string | null>(null)

// Computed getters for available result sets
const keywordSets = computed(() => {
  if (!analysisData.value?.keywords) return []
  return Object.keys(analysisData.value.keywords)
})

const emotionSets = computed(() => {
  if (!analysisData.value?.emotions) return []
  return Object.keys(analysisData.value.emotions)
})

const layoutSets = computed(() => {
  if (!analysisData.value?.layout) return []
  return Object.keys(analysisData.value.layout)
})

const entitySets = computed(() => {
  if (!analysisData.value?.entities) return []
  return Object.keys(analysisData.value.entities)
})

// Computed getters for selected data
const displayedKeywords = computed(() => {
  if (!selectedKeywords.value || !analysisData.value?.keywords) return []
  return analysisData.value.keywords[selectedKeywords.value] || []
})

const displayedEmotions = computed(() => {
  if (!selectedEmotions.value || !analysisData.value?.emotions) return []
  return analysisData.value.emotions[selectedEmotions.value] || []
})

const displayedLayout = computed(() => {
  if (!selectedLayout.value || !analysisData.value?.layout) return []
  return analysisData.value.layout[selectedLayout.value] || []
})

// Computed stats for layout regions by type
const layoutStats = computed(() => {
  const stats: Record<string, number> = {}
  displayedLayout.value.forEach((region: any) => {
    const className = region.class_name || 'Unknown'
    stats[className] = (stats[className] || 0) + 1
  })
  return Object.entries(stats)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

// Computed stats for emotions by type
const emotionStats = computed(() => {
  const stats: Record<string, number> = {}
  displayedEmotions.value.forEach((em: any) => {
    const emotion = em.label || em.emotion || 'Unknown'
    stats[emotion] = (stats[emotion] || 0) + 1
  })
  return Object.entries(stats)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

const displayedEntities = computed(() => {
  if (!selectedEntities.value || !analysisData.value?.entities) return []
  return analysisData.value.entities[selectedEntities.value] || []
})

// Computed stats for entities by type
const entityStats = computed(() => {
  const stats: Record<string, number> = {}
  displayedEntities.value.forEach((ent: any) => {
    const type = (ent.entity_type || ent.label || 'Unknown').toLowerCase()
    stats[type] = (stats[type] || 0) + 1
  })
  return Object.entries(stats)
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => a.name.localeCompare(b.name))
})

// Emotion icon and color mapping
const emotionIcons: Record<string, any> = {
  'Sadness': Frown,
  'Love': HeartHandshake,
  'Joy': Smile,
  'Fear': Skull,
  'Anger': Flame,
  'Agitation': Zap,
}

const emotionColors: Record<string, string> = {
  'Joy': '#FFD700',      // Gold/Yellow - happiness, sunshine
  'Love': '#FF69B4',     // Hot Pink - romance, affection
  'Anger': '#DC143C',    // Crimson Red - rage, intensity
  'Fear': '#9370DB',     // Medium Purple - anxiety, unease
  'Sadness': '#4682B4',  // Steel Blue - melancholy, tears
  'Agitation': '#FF4500', // Orange Red - restlessness, alarm
}

// Entity type colors and icons
const entityTypeColors: Record<string, string> = {
  person: '#2E5EFF',      // Vibrant Blue
  organization: '#FF3333', // Vivid Red
  location: '#00E676',     // Bright Green
  date: '#9C27FF',        // Vivid Purple
  event: '#FF9100',       // Bright Orange
  misc: '#00E5FF',        // Bright Cyan
}

const entityTypeIcons: Record<string, any> = {
  person: User,
  organization: Building2,
  location: MapPin,
  event: Calendar,
  date: Calendar,
  misc: Tag,
}

// Layout class colors (from imageAnnotation.ts)
const layoutClassColors: Record<string, string> = {
  'Text': '#FF4444',
  'Picture': '#44FF44',
  'Section-header': '#4444FF',
  'Table': '#FFFF44',
  'Page-header': '#FF44FF',
  'Page-footer': '#44FFFF',
  'Caption': '#FFA500',
  'List': '#800080',
  'Title': '#FF1493',
  'Figure': '#00CED1',
  'Formula': '#FFD700',
}

// Group emotions by text line
const emotionsByLine = computed(() => {
  const grouped: Record<string, any[]> = {}
  displayedEmotions.value.forEach((em: any) => {
    const lineId = em.line_id || 'unknown'
    if (!grouped[lineId]) {
      grouped[lineId] = []
    }
    grouped[lineId].push(em)
  })
  return grouped
})

// Helper functions to get counts for each set
function getKeywordsCount(setName: string): number {
  return analysisData.value?.keywords?.[setName]?.length || 0
}

function getEmotionsCount(setName: string): number {
  return analysisData.value?.emotions?.[setName]?.length || 0
}

function getLayoutCount(setName: string): number {
  return analysisData.value?.layout?.[setName]?.length || 0
}

function getEntitiesCount(setName: string): number {
  return analysisData.value?.entities?.[setName]?.length || 0
}

watch(() => props.isOpen, async (isOpen) => {
  if (isOpen && props.pageId && props.sourceName) {
    await loadAnalysis()
  }
})

// Also watch for pageId changes (e.g., when navigating between pages)
watch(() => props.pageId, async (pageId) => {
  if (props.isOpen && pageId && props.sourceName) {
    await loadAnalysis()
  }
})

// Auto-select first available result set when data loads
watch(() => analysisData.value, (data) => {
  if (data) {
    if (keywordSets.value.length > 0 && !selectedKeywords.value) {
      selectedKeywords.value = keywordSets.value[0]
    }
    if (emotionSets.value.length > 0 && !selectedEmotions.value) {
      selectedEmotions.value = emotionSets.value[0]
    }
    if (layoutSets.value.length > 0 && !selectedLayout.value) {
      selectedLayout.value = layoutSets.value[0]
    }
    if (entitySets.value.length > 0 && !selectedEntities.value) {
      selectedEntities.value = entitySets.value[0]
    }
  }
})

async function loadAnalysis() {
  if (!props.pageId || !props.sourceName) return
  
  loading.value = true
  try {
    const response = await api.get(`/data/${props.sourceName}/page-analysis/${props.pageId}`)
    analysisData.value = response.data
  } catch (error) {
    console.error('Failed to load analysis:', error)
    console.error('Failed to load analysis:', error)
  } finally {
    loading.value = false
  }
}

function close() {
  emit('close')
}
</script>

<template>
  <!-- Sidebar mode -->
  <div v-if="props.sidebarMode" class="h-full flex flex-col">
    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-3 space-y-3 bg-card">
      <div v-if="loading" class="text-center py-8">
        <p class="text-muted-foreground text-sm">Loading...</p>
      </div>

      <div v-else-if="analysisData" class="space-y-0">
            <!-- Emotions -->
            <div v-if="emotionSets.length > 0">
              <div
                @click="emotionsExpanded = !emotionsExpanded"
                class="flex items-center justify-between p-4 cursor-pointer hover:bg-accent transition-colors"
              >
                <div class="flex items-center gap-2">
                  <component :is="emotionsExpanded ? ChevronDown : ChevronRight" class="h-4 w-4" />
                  <Smile class="h-5 w-5" />
                  <h3 class="text-lg font-semibold">Emotions</h3>
                </div>
                <select
                  v-model="selectedEmotions"
                  @click.stop
                  class="px-3 py-1 text-sm border rounded-lg bg-background w-56 truncate"
                >
                  <option v-for="set in emotionSets" :key="set" :value="set" class="truncate">
                    {{ set }} ({{ getEmotionsCount(set) }})
                  </option>
                </select>
              </div>
              <div v-if="emotionsExpanded" class="px-4 pb-4 pt-2 space-y-3">
                <div class="border-t mb-3"></div>
                <p v-if="selectedEmotions" class="text-xs text-muted-foreground">{{ selectedEmotions }} ({{ getEmotionsCount(selectedEmotions) }} detections)</p>
                <!-- Aggregated Stats -->
                <div class="grid grid-cols-3 gap-2">
                  <div
                    v-for="stat in emotionStats"
                    :key="stat.name"
                    class="p-2 rounded-lg"
                    :style="{ backgroundColor: emotionColors[stat.name] + '20' }"
                  >
                    <div class="flex items-center justify-center gap-2 mb-1">
                      <component 
                        :is="emotionIcons[stat.name] || Smile" 
                        class="h-5 w-5" 
                      />
                      <div class="text-2xl font-bold">
                        {{ stat.count }}
                      </div>
                    </div>
                    <div class="text-xs text-muted-foreground text-center">{{ stat.name }}</div>
                  </div>
                </div>
                
                <!-- Detailed Listing (nested collapsible) -->
                <div class="border-t pt-3">
                  <button
                    @click="emotionDetailsExpanded = !emotionDetailsExpanded"
                    class="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors"
                  >
                    <component :is="emotionDetailsExpanded ? ChevronDown : ChevronRight" class="h-3 w-3" />
                    Detailed Text Lines ({{ Object.keys(emotionsByLine).length }} lines)
                  </button>
                  <div v-if="emotionDetailsExpanded" class="mt-3 space-y-2">
                    <div
                      v-for="(emotions, lineId) in emotionsByLine"
                      :key="lineId"
                      class="p-3 bg-accent rounded-lg"
                    >
                      <div class="flex flex-wrap gap-2 mb-2">
                        <div
                          v-for="(em, idx) in emotions"
                          :key="idx"
                          class="flex items-center gap-1 px-2 py-1 rounded text-xs"
                          :style="{ backgroundColor: emotionColors[em.label || em.emotion] + '20' }"
                        >
                          <component 
                            :is="emotionIcons[em.label || em.emotion] || Smile" 
                            class="h-3 w-3"
                          />
                          <span class="font-medium">
                            {{ em.label || em.emotion }}
                          </span>
                          <span class="text-muted-foreground">{{ (em.score * 100).toFixed(1) }}%</span>
                        </div>
                      </div>
                      <p v-if="emotions[0]?.text" class="text-sm text-muted-foreground">{{ emotions[0].text }}</p>
                    </div>
                  </div>
                </div>
              </div>
              <div class="border-t my-3"></div>
            </div>

            <!-- Entities -->
            <div v-if="entitySets.length > 0">
              <div
                @click="entitiesExpanded = !entitiesExpanded"
                class="flex items-center justify-between p-4 cursor-pointer hover:bg-accent transition-colors"
              >
                <div class="flex items-center gap-2">
                  <component :is="entitiesExpanded ? ChevronDown : ChevronRight" class="h-4 w-4" />
                  <Users class="h-5 w-5" />
                  <h3 class="text-lg font-semibold">Entities</h3>
                </div>
                <select
                  v-model="selectedEntities"
                  @click.stop
                  class="px-3 py-1 text-sm border rounded-lg bg-background w-56 truncate"
                >
                  <option v-for="set in entitySets" :key="set" :value="set" class="truncate">
                    {{ set }} ({{ getEntitiesCount(set) }})
                  </option>
                </select>
              </div>
              <div v-if="entitiesExpanded" class="p-4 pt-0 space-y-3">
                <div class="border-t mb-3"></div>
                <p v-if="selectedEntities" class="text-xs text-muted-foreground">{{ selectedEntities }} ({{ getEntitiesCount(selectedEntities) }} entities)</p>
                
                <!-- Aggregated Stats -->
                <div class="grid grid-cols-3 gap-2">
                  <div
                    v-for="stat in entityStats"
                    :key="stat.name"
                    class="p-2 rounded-lg"
                    :style="{ backgroundColor: (entityTypeColors[stat.name] || '#888') + '20' }"
                  >
                    <div class="flex items-center justify-center gap-2 mb-1">
                      <component 
                        :is="entityTypeIcons[stat.name] || Tag" 
                        class="h-5 w-5" 
                      />
                      <div class="text-2xl font-bold">
                        {{ stat.count }}
                      </div>
                    </div>
                    <div class="text-xs text-muted-foreground text-center">{{ stat.name }}</div>
                  </div>
                </div>

                <!-- Entity List -->
                <div class="border-t pt-3 space-y-2">
                <div
                  v-for="(ent, idx) in displayedEntities.slice(0, 20)"
                  :key="idx"
                  class="p-3 bg-accent rounded-lg"
                >
                  <div class="flex justify-between items-center gap-2">
                    <span class="font-medium">{{ ent.entity_text || ent.text }}</span>
                    <span 
                      class="flex items-center gap-1 text-sm px-2 py-1 rounded"
                      :style="{ backgroundColor: (entityTypeColors[(ent.entity_type || ent.label)?.toLowerCase()] || '#888') + '20' }"
                    >
                      <component 
                        :is="entityTypeIcons[(ent.entity_type || ent.label)?.toLowerCase()] || Tag" 
                        class="h-3 w-3"
                      />
                      <span class="font-medium">{{ ent.entity_type || ent.label }}</span>
                    </span>
                  </div>
                  <p v-if="ent.confidence || ent.score" class="text-sm text-muted-foreground mt-1">
                    Confidence: {{ ((ent.confidence || ent.score) * 100).toFixed(1) }}%
                  </p>
                </div>
                </div>
              </div>
              <div class="border-t my-3"></div>
            </div>

            <!-- Keywords -->
            <div v-if="keywordSets.length > 0">
              <div
                @click="keywordsExpanded = !keywordsExpanded"
                class="flex items-center justify-between p-4 cursor-pointer hover:bg-accent transition-colors"
              >
                <div class="flex items-center gap-2">
                  <component :is="keywordsExpanded ? ChevronDown : ChevronRight" class="h-4 w-4" />
                  <Tag class="h-5 w-5" />
                  <h3 class="text-lg font-semibold">Keywords</h3>
                </div>
                <select
                  v-model="selectedKeywords"
                  @click.stop
                  class="px-3 py-1 text-sm border rounded-lg bg-background w-56 truncate"
                >
                  <option v-for="set in keywordSets" :key="set" :value="set" class="truncate">
                    {{ set }} ({{ getKeywordsCount(set) }})
                  </option>
                </select>
              </div>
              <div v-if="keywordsExpanded" class="px-4 pb-4 pt-2">
                <div class="border-t mb-3"></div>
                <p v-if="selectedKeywords" class="text-xs text-muted-foreground mb-3">{{ selectedKeywords }} ({{ getKeywordsCount(selectedKeywords) }} keywords)</p>
                <div class="flex flex-wrap gap-2">
                  <span
                    v-for="(kw, idx) in displayedKeywords.slice(0, 30)"
                    :key="idx"
                    class="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm"
                  >
                    {{ kw.keyword }}
                    <span v-if="kw.score" class="text-xs opacity-70">({{ kw.score.toFixed(2) }})</span>
                  </span>
                </div>
              </div>
              <div class="border-t my-3"></div>
            </div>

            <!-- Layout Regions -->
            <div v-if="layoutSets.length > 0">
              <div
                @click="layoutExpanded = !layoutExpanded"
                class="flex items-center justify-between p-4 cursor-pointer hover:bg-accent transition-colors"
              >
                <div class="flex items-center gap-2">
                  <component :is="layoutExpanded ? ChevronDown : ChevronRight" class="h-4 w-4" />
                  <LayoutDashboard class="h-5 w-5" />
                  <h3 class="text-lg font-semibold">Layout Regions</h3>
                </div>
                <select
                  v-model="selectedLayout"
                  @click.stop
                  class="px-3 py-1 text-sm border rounded-lg bg-background w-56 truncate"
                >
                  <option v-for="set in layoutSets" :key="set" :value="set" class="truncate">
                    {{ set }} ({{ getLayoutCount(set) }})
                  </option>
                </select>
              </div>
              <div v-if="layoutExpanded" class="px-4 pb-4 pt-2 space-y-3">
                <div class="border-t mb-3"></div>
                <p v-if="selectedLayout" class="text-xs text-muted-foreground">{{ selectedLayout }} ({{ getLayoutCount(selectedLayout) }} regions)</p>
                <!-- Aggregated Stats -->
                <div class="grid grid-cols-3 gap-2">
                  <div
                    v-for="stat in layoutStats"
                    :key="stat.name"
                    class="p-2 rounded-lg"
                    :style="{ backgroundColor: (layoutClassColors[stat.name] || '#888') + '20' }"
                  >
                    <div class="flex items-center justify-center gap-2 mb-1">
                      <LayoutDashboard class="h-5 w-5" />
                      <div class="text-2xl font-bold">{{ stat.count }}</div>
                    </div>
                    <div class="text-xs text-muted-foreground text-center">{{ stat.name }}</div>
                  </div>
                </div>
                
                <!-- Detailed Listing (nested collapsible) -->
                <div class="border-t pt-3">
                  <button
                    @click="layoutDetailsExpanded = !layoutDetailsExpanded"
                    class="flex items-center gap-2 text-sm font-medium hover:text-primary transition-colors"
                  >
                    <component :is="layoutDetailsExpanded ? ChevronDown : ChevronRight" class="h-3 w-3" />
                    Detailed Regions
                  </button>
                  <div v-if="layoutDetailsExpanded" class="mt-3 grid grid-cols-3 gap-2">
                    <div
                      v-for="(region, idx) in displayedLayout"
                      :key="idx"
                      class="p-2 bg-accent rounded-lg"
                    >
                      <div class="flex flex-col gap-1">
                        <div class="flex items-center justify-between gap-1">
                          <span 
                            class="flex items-center gap-1 px-2 py-1 rounded text-xs font-medium"
                            :style="{ backgroundColor: (layoutClassColors[region.class_name] || '#888') + '20' }"
                          >
                            <LayoutDashboard class="h-3 w-3" />
                            <span class="truncate">{{ region.class_name }}</span>
                          </span>
                          <span class="text-xs text-muted-foreground whitespace-nowrap">{{ (region.confidence * 100).toFixed(0) }}%</span>
                        </div>
                        <p v-if="region.text_content" class="text-xs text-muted-foreground line-clamp-2">
                          {{ region.text_content }}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div class="border-t my-3"></div>
            </div>

            <!-- No results -->
            <div v-if="keywordSets.length === 0 && emotionSets.length === 0 && layoutSets.length === 0 && entitySets.length === 0">
              <p class="text-center text-muted-foreground py-12">
                No analysis results available for this page
              </p>
            </div>
          </div>
      </div>
    </div>
</template>
