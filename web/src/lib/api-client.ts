/**
 * API Client for LARUN.SPACE Backend
 *
 * Handles communication with FastAPI backend
 */

import axios, { AxiosInstance } from 'axios'
import { InferenceResult } from './supabase'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Log API URL for debugging
console.log('API Client initialized with URL:', API_URL)

class APIClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_URL,
      timeout: 120000, // 2 minutes for long analyses
      headers: {
        'Content-Type': 'application/json',
      },
    })
    console.log('API Client baseURL:', this.client.defaults.baseURL)
  }

  setAuthToken(token: string) {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`
  }

  clearAuthToken() {
    delete this.client.defaults.headers.common['Authorization']
  }

  // TinyML Analysis — always routes to Next.js API (relative URL)
  async analyzeTinyML(
    file: File,
    modelId: string,
    userId: string
  ): Promise<InferenceResult> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_id', modelId)
    formData.append('user_id', userId)

    // Use relative URL so it always hits the Next.js API route regardless of
    // whether NEXT_PUBLIC_API_URL is set (avoids "Backend API not available")
    const response = await axios.post('/api/tinyml/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 30000,
    })

    return response.data
  }

  // Get available models
  async getModels() {
    const response = await axios.get('/api/tinyml/models')
    return response.data
  }

  // Get user quota
  async getUserQuota(userId: string) {
    const response = await this.client.get(`/api/user/quota/${userId}`)
    return response.data
  }

  // Health check
  async healthCheck() {
    const response = await this.client.get('/health')
    return response.data
  }

  // Existing endpoints (BLS, transit fitting, etc.)
  async analyzeBLS(data: {
    time: number[]
    flux: number[]
    flux_err?: number[]
    min_period?: number
    max_period?: number
    min_snr?: number
  }) {
    const response = await this.client.post('/analyze/bls', { data })
    return response.data
  }

  async analyzeTransitFit(data: {
    time: number[]
    flux: number[]
    flux_err?: number[]
    period: number
    t0?: number
    stellar_teff?: number
  }) {
    const response = await this.client.post('/analyze/fit', { data })
    return response.data
  }

  async classifyStellar(params: {
    teff: number
    logg?: number
    metallicity?: number
  }) {
    const response = await this.client.post('/stellar/classify', params)
    return response.data
  }

  async calculatePlanetRadius(params: {
    depth_ppm: number
    stellar_radius: number
    period?: number
    stellar_mass?: number
    stellar_teff?: number
    stellar_luminosity?: number
  }) {
    const response = await this.client.post('/planet/radius', params)
    return response.data
  }

  async calculateHabitableZone(params: {
    stellar_teff: number
    stellar_luminosity: number
  }) {
    const response = await this.client.post('/planet/hz', params)
    return response.data
  }

  async runPipeline(params: {
    target: string
    quick_mode?: boolean
  }) {
    const response = await this.client.post('/pipeline', params)
    return response.data
  }
}

export const apiClient = new APIClient()

// Model registry
export const TINYML_MODELS = [
  {
    id: 'EXOPLANET-001',
    name: 'Exoplanet Transit Detector',
    description: 'Detects planetary transits in light curves',
    input_length: 1024,
    classes: ['noise', 'stellar_signal', 'planetary_transit', 'eclipsing_binary', 'instrument_artifact', 'unknown_anomaly'],
    accuracy: 0.818,
    size_kb: 26,
  },
  {
    id: 'VSTAR-001',
    name: 'Variable Star Classifier',
    description: 'Classifies variable star types',
    input_length: 512,
    classes: ['cepheid', 'rr_lyrae', 'delta_scuti', 'eclipsing_binary', 'rotational', 'irregular', 'constant'],
    accuracy: 0.952,
    size_kb: 27,
  },
  {
    id: 'FLARE-001',
    name: 'Stellar Flare Detector',
    description: 'Detects and classifies stellar flares',
    input_length: 256,
    classes: ['no_flare', 'weak_flare', 'moderate_flare', 'strong_flare', 'superflare'],
    accuracy: 0.976,
    size_kb: 13,
  },
  {
    id: 'ASTERO-001',
    name: 'Asteroseismology Analyzer',
    description: 'Analyzes stellar oscillations',
    input_length: 512,
    classes: ['solar_like', 'classical_pulsator', 'hybrid', 'non_pulsating'],
    accuracy: 0.937,
    size_kb: 16,
  },
  {
    id: 'SUPERNOVA-001',
    name: 'Supernova Detector',
    description: 'Detects supernova and transient events',
    input_length: 128,
    classes: ['no_transient', 'supernova_ia', 'supernova_ii', 'tde', 'agn'],
    accuracy: 0.944,
    size_kb: 24,
  },
  {
    id: 'GALAXY-001',
    name: 'Galaxy Morphology Classifier',
    description: 'Classifies galaxy morphologies',
    input_length: 4096,
    classes: ['elliptical', 'spiral', 'irregular', 'merger'],
    accuracy: 0.963,
    size_kb: 26,
  },
  {
    id: 'SPECTYPE-001',
    name: 'Spectral Type Classifier',
    description: 'Classifies stellar spectral types',
    input_length: 8,
    classes: ['O', 'B', 'A', 'F', 'G', 'K', 'M'],
    accuracy: 0.981,
    size_kb: 5,
  },
  {
    id: 'MICROLENS-001',
    name: 'Microlensing Detector',
    description: 'Detects gravitational microlensing events',
    input_length: 512,
    classes: ['no_event', 'single_lens', 'binary_lens', 'planetary'],
    accuracy: 0.891,
    size_kb: 26,
  },
]

export const getModelById = (modelId: string) => {
  return TINYML_MODELS.find(m => m.id === modelId)
}

export const getModelAccuracy = (modelId: string): number => {
  const model = getModelById(modelId)
  return model?.accuracy || 0
}
