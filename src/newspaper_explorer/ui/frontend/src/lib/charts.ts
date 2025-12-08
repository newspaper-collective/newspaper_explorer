/**
 * Composables for Apache ECharts
 */

import type { EChartsOption } from 'echarts'
import 'echarts-wordcloud'

/**
 * Default theme configuration for all charts
 */
export function useChartTheme() {
  return {
    color: [
      '#2E5EFF', // Vibrant Blue
      '#FF3333', // Vivid Red
      '#00E676', // Bright Green
      '#9C27FF', // Vivid Purple
      '#FF9100', // Bright Orange
      '#00E5FF', // Bright Cyan
      '#FF1744', // Bright Pink/Red
      '#76FF03', // Neon Green
      '#E040FB', // Bright Magenta
      '#FFEA00', // Bright Yellow
    ],
    backgroundColor: 'transparent',
    textStyle: {
      fontFamily: 'Inter, system-ui, sans-serif',
    },
  }
}

/**
 * Common chart options
 */
export function useCommonChartOptions(): Partial<EChartsOption> {
  const theme = useChartTheme()

  return {
    color: theme.color,
    backgroundColor: theme.backgroundColor,
    textStyle: theme.textStyle,
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '8%',
      containLabel: true,
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.95)',
      borderColor: '#ddd',
      borderWidth: 1,
      textStyle: {
        color: '#333',
      },
      axisPointer: {
        type: 'shadow',
        shadowStyle: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
    toolbox: {
      feature: {
        saveAsImage: {
          title: 'Save as Image',
          pixelRatio: 2,
        },
        dataZoom: {
          title: {
            zoom: 'Zoom',
            back: 'Reset Zoom',
          },
        },
      },
      iconStyle: {
        borderColor: '#666',
      },
    },
  }
}

/**
 * Create a bar chart configuration
 */
export function useBarChart(
  data: { name: string; value: number }[],
  options: Partial<EChartsOption> = {}
): EChartsOption {
  const common = useCommonChartOptions()

  return {
    ...common,
    ...options,
    xAxis: {
      type: 'category',
      data: data.map((d) => d.name),
      axisLabel: {
        interval: 0,
        rotate: data.length > 10 ? 45 : 0,
      },
      ...options.xAxis,
    },
    yAxis: {
      type: 'value',
      ...options.yAxis,
    },
    series: [
      {
        type: 'bar',
        data: data.map((d) => d.value),
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.3)',
          },
        },
        ...(options.series?.[0] || {}),
      },
    ],
  }
}

/**
 * Create a line chart configuration
 */
export function useLineChart(
  data: { name: string; value: number }[],
  options: Partial<EChartsOption> = {}
): EChartsOption {
  const common = useCommonChartOptions()

  return {
    ...common,
    ...options,
    xAxis: {
      type: 'category',
      data: data.map((d) => d.name),
      boundaryGap: false,
      ...options.xAxis,
    },
    yAxis: {
      type: 'value',
      ...options.yAxis,
    },
    series: [
      {
        type: 'line',
        data: data.map((d) => d.value),
        smooth: true,
        lineStyle: {
          width: 2,
        },
        areaStyle: {
          opacity: 0.2,
        },
        emphasis: {
          focus: 'series',
        },
        ...(options.series?.[0] || {}),
      },
    ],
  }
}

/**
 * Create a pie chart configuration
 */
export function usePieChart(
  data: { name: string; value: number }[],
  options: Partial<EChartsOption> = {}
): EChartsOption {
  const common = useCommonChartOptions()

  return {
    ...common,
    ...options,
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)',
    },
    legend: {
      orient: 'horizontal',
      bottom: 10,
      ...options.legend,
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 4,
          borderColor: '#fff',
          borderWidth: 2,
        },
        label: {
          show: false,
          position: 'center',
        },
        emphasis: {
          label: {
            show: true,
            fontSize: 20,
            fontWeight: 'bold',
          },
        },
        labelLine: {
          show: false,
        },
        data: data,
        ...(options.series?.[0] || {}),
      },
    ],
  }
}

/**
 * Create a timeline chart (stacked area or bar)
 */
export function useTimelineChart(
  data: Record<string, { date: string; value: number }[]>,
  chartType: 'bar' | 'line' = 'bar',
  options: Partial<EChartsOption> & { _colors?: string[] } = {}
): EChartsOption {
  const common = useCommonChartOptions()

  // Get all unique dates
  const dates = Array.from(
    new Set(Object.values(data).flatMap((series) => series.map((d) => d.date)))
  ).sort()

  // Extract custom _colors option if provided
  const customColors = options._colors
  const { _colors, ...chartOptions } = options

  // Build series for each category
  const series = Object.entries(data).map(([name, values], index) => {
    const valueMap = new Map(values.map((v) => [v.date, v.value]))
    const color = customColors?.[index]
    
    return {
      name,
      type: chartType,
      stack: chartType === 'bar' ? 'total' : undefined,
      data: dates.map((date) => valueMap.get(date) || 0),
      smooth: chartType === 'line',
      areaStyle: chartType === 'line' ? { opacity: 0.6 } : undefined,
      itemStyle:
        chartType === 'bar'
          ? {
              borderRadius: [2, 2, 0, 0],
              color: color,
            }
          : {
              color: color,
            },
    }
  })

  return {
    ...common,
    ...chartOptions,
    legend: {
      data: Object.keys(data),
      bottom: 0,
      ...chartOptions.legend,
    },
    grid: {
      ...common.grid,
      bottom: '15%', // More space for legend
    },
    xAxis: {
      type: 'category',
      data: dates,
      axisLabel: {
        rotate: 45,
      },
      ...chartOptions.xAxis,
    },
    yAxis: {
      type: 'value',
      ...chartOptions.yAxis,
    },
    series,
  }
}

/**
 * Create a wordcloud chart configuration
 */
export function useWordCloud(
  data: { name: string; value: number }[],
  options: Partial<EChartsOption> = {}
): EChartsOption {
  const common = useCommonChartOptions()

  return {
    ...common,
    ...options,
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c}',
    },
    series: [
      {
        type: 'wordCloud',
        shape: 'square',  // Classic rectangular wordcloud
        keepAspect: false,
        left: 'center',
        top: 'center',
        width: '90%',
        height: '90%',
        right: null,
        bottom: null,
        sizeRange: [16, 80],  // Even larger size range for better visibility
        rotationRange: [0, 0],  // No rotation for classic look
        rotationStep: 0,
        gridSize: 8,
        drawOutOfBound: false,
        layoutAnimation: true,
        textStyle: {
          fontFamily: 'Inter, system-ui, sans-serif',
          fontWeight: 'bold',
        },
        emphasis: {
          focus: 'self',
          textStyle: {
            textShadowBlur: 10,
            textShadowColor: '#333',
          },
        },
        data: data,
        ...(options.series?.[0] || {}),
      } as any, // Type assertion needed for wordCloud
    ],
  }
}
