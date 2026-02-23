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

  // Spectral type classification from colour indices (SPECTYPE-001)
  async analyzeSpectralType(
    indices: { bv?: number | null; vr?: number | null; bp_rp?: number | null; jh?: number | null; hk?: number | null },
    userId: string
  ): Promise<InferenceResult> {
    const response = await axios.post('/api/tinyml/analyze-spectype', {
      ...indices,
      user_id: userId,
    });
    return response.data;
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
    description: 'Detects and classifies transit-shaped dips in photometric light curves — distinguishes true planetary transits from eclipsing binaries, stellar signals, and instrument artefacts.',
    use_case: 'Find exoplanet candidates in TESS / Kepler photometry',
    data_source: 'Photometry — TESS, Kepler, K2',
    category: 'exoplanet',
    input_length: 1024,
    classes: ['noise', 'stellar_signal', 'planetary_transit', 'eclipsing_binary', 'instrument_artifact', 'unknown_anomaly'],
    accuracy: 0.818,
    size_kb: 26,
  },
  {
    id: 'VSTAR-001',
    name: 'Variable Star Classifier',
    description: 'Identifies the pulsation type of variable stars from their light-curve shape and variability — covering Cepheids, RR Lyrae, Delta Scuti, eclipsing binaries, rotational modulators, and irregular variables.',
    use_case: 'Classify what type of variable star you are observing',
    data_source: 'Photometry — TESS, ASAS-SN, ZTF, OGLE',
    category: 'stellar',
    input_length: 512,
    classes: ['cepheid', 'rr_lyrae', 'delta_scuti', 'eclipsing_binary', 'rotational', 'irregular', 'constant'],
    accuracy: 0.952,
    size_kb: 27,
  },
  {
    id: 'FLARE-001',
    name: 'Stellar Flare Detector',
    description: 'Detects impulsive brightening events caused by magnetic reconnection on stellar surfaces and grades them from weak micro-flares to extreme superflares.',
    use_case: 'Detect and grade flare events in high-cadence light curves',
    data_source: 'Photometry — TESS (2-min cadence), Kepler SC',
    category: 'stellar',
    input_length: 256,
    classes: ['no_flare', 'weak_flare', 'moderate_flare', 'strong_flare', 'superflare'],
    accuracy: 0.976,
    size_kb: 13,
  },
  {
    id: 'ASTERO-001',
    name: 'Asteroseismology Analyzer',
    description: 'Characterises stellar oscillation regimes from photometric variability — separates solar-like stochastic oscillators from classical heat-engine pulsators and hybrid stars.',
    use_case: 'Identify stellar oscillation type for seismic analysis',
    data_source: 'Photometry — Kepler, TESS (2-min or FFI)',
    category: 'stellar',
    input_length: 512,
    classes: ['solar_like', 'classical_pulsator', 'hybrid', 'non_pulsating'],
    accuracy: 0.937,
    size_kb: 16,
  },
  {
    id: 'SUPERNOVA-001',
    name: 'Supernova & Transient Detector',
    description: 'Identifies extragalactic transient events in multi-epoch photometry — classifies Type Ia and core-collapse supernovae, tidal disruption events (TDEs), and active galactic nuclei variability.',
    use_case: 'Detect and type supernovae or other transient events',
    data_source: 'Photometry — ZTF, ATLAS, Rubin/LSST alerts',
    category: 'transient',
    input_length: 128,
    classes: ['no_transient', 'supernova_ia', 'supernova_ii', 'tde', 'agn'],
    accuracy: 0.944,
    size_kb: 24,
  },
  {
    id: 'MICROLENS-001',
    name: 'Microlensing Detector',
    description: 'Finds gravitational microlensing events — smooth single-lens Paczyński curves, caustic-crossing binary-lens events, and short-duration planetary anomalies on top of a stellar lens.',
    use_case: 'Search for microlensing events in Galactic bulge surveys',
    data_source: 'Photometry — OGLE, KMTNet, MOA, Roman (future)',
    category: 'transient',
    input_length: 512,
    classes: ['no_event', 'single_lens', 'binary_lens', 'planetary'],
    accuracy: 0.891,
    size_kb: 26,
  },
  {
    id: 'GALAXY-001',
    name: 'Galaxy Morphology Classifier',
    description: 'Classifies galaxy morphology from photometric light-profile features — ellipticals, spirals, irregulars, and ongoing mergers.',
    use_case: 'Classify galaxy morphology from photometric data',
    data_source: 'Photometry — SDSS, HST, Euclid, JWST',
    category: 'galactic',
    input_length: 4096,
    classes: ['elliptical', 'spiral', 'irregular', 'merger'],
    accuracy: 0.963,
    size_kb: 26,
  },
  {
    id: 'SPECTYPE-001',
    name: 'Spectral Type Classifier',
    description: 'Assigns an MK spectral type (O through M) to a star from a compact set of photometric colour indices or low-resolution spectral features.',
    use_case: 'Determine stellar spectral type from colours or spectra',
    data_source: 'Photometry / Spectra — Gaia DR3, 2MASS, SDSS',
    category: 'stellar',
    input_length: 8,
    classes: ['O', 'B', 'A', 'F', 'G', 'K', 'M'],
    accuracy: 0.981,
    size_kb: 5,
  },
]

export const getModelById = (modelId: string) => {
  return TINYML_MODELS.find(m => m.id === modelId)
}

export const getModelAccuracy = (modelId: string): number => {
  const model = getModelById(modelId)
  return model?.accuracy || 0
}
