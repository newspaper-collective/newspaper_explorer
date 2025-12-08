import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 30000,
  paramsSerializer: {
    indexes: null, // Use param=value1&param=value2 format for arrays (FastAPI compatible)
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

export default api
