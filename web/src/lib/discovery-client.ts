/**
 * Discovery API Client — LARUN v2
 * TypeScript client for /api/v2/* endpoints (Citizen Discovery Engine)
 */

const API_V2 = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v2`
  : '/api/v2'

// ── Types ──────────────────────────────────────────────────────────────────

export interface DiscoveryTarget {
  ra: number
  dec: number
  name?: string
}

export interface ClassificationResult {
  model_id: string
  label: string
  confidence: number
  probabilities?: Record<string, number>
  inference_ms?: number
  features?: Record<string, number>
  error?: string
}

export interface CatalogMatch {
  known: boolean
  novelty_score: number
  matches: Array<{
    catalog: string
    name?: string
    distance_arcsec: number
    object_type?: string
  }>
}

export interface ConsensusResult {
  consensus_label: string
  consensus_confidence: number
  is_variable: boolean
  anomaly_detected: boolean
  blend_detected: boolean
  agreement_count: number
}

export interface DiscoveryCandidate {
  target: DiscoveryTarget
  light_curve_meta: {
    n_points: number
    baseline_days?: number
    cadence_min?: number
  }
  classifications: Record<string, ClassificationResult>
  catalog_match: CatalogMatch
  consensus: ConsensusResult
  priority: number
  source: string
  is_candidate: boolean
  novelty_score: number
  period_days?: number
  period_confidence?: number
  period_type?: string
}

export interface DiscoveryReport {
  candidates: DiscoveryCandidate[]
  known: DiscoveryCandidate[]
  anomalies: DiscoveryCandidate[]
  stats: {
    total: number
    candidates: number
    known: number
    analyzed: number
    elapsed_seconds: number
  }
  meta: {
    ra: number
    dec: number
    radius_deg: number
    sources: string[]
    models_used: string
  }
}

export interface LeaderboardEntry {
  rank: number
  user_id: string
  verified_discoveries: number
  points: number
  title: string
  last_discovery?: string
}

export interface Leaderboard {
  rankings: LeaderboardEntry[]
  total_users: number
  total_discoveries: number
  period: string
}

export interface UserStats {
  user_id: string
  total_submissions: number
  verified_discoveries: number
  points: number
  rank: string
  recent_discoveries: DiscoveryCandidate[]
}

export interface VerificationResult {
  discovery_id: string
  verdict: string
  confirmations: number
  rejections: number
  new_status: string
}

export interface DiscoverRequest {
  ra: number
  dec: number
  radius_deg?: number
  sources?: string[]
  models?: string | string[]
}

// ── Client ─────────────────────────────────────────────────────────────────

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_V2}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`API error ${res.status}: ${err}`)
  }
  return res.json()
}

async function get<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = new URL(`${API_V2}${path}`, typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000')
  if (params) {
    Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, String(v)))
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`API error ${res.status}: ${err}`)
  }
  return res.json()
}

// ── Discovery ──────────────────────────────────────────────────────────────

export const discoveryClient = {
  /** Run the full Citizen Discovery Engine on a sky region */
  discover(req: DiscoverRequest): Promise<DiscoveryReport> {
    return post('/discover', req)
  },

  /** List all stored discovery candidates */
  getCandidates(limit = 50): Promise<{ candidates: DiscoveryCandidate[]; total: number }> {
    return get('/discover/candidates', { limit })
  },

  /** Submit a peer verification vote */
  verify(discoveryId: string, verdict: 'confirm' | 'reject', verifierId: string): Promise<VerificationResult> {
    return post(`/discover/verify/${discoveryId}`, { verdict, verifier_id: verifierId })
  },

  /** Parse a natural-language query (Claude) and run discovery */
  discoverNL(query: string): Promise<DiscoveryReport> {
    return post('/discover/nl', { query })
  },

  /** Run all Layer-2 models on a pre-fetched light curve dict */
  runFederation(lightCurve: {
    times: number[]
    flux: number[]
    flux_err?: number[]
  }): Promise<{
    results: Record<string, ClassificationResult>
    consensus: ConsensusResult
    summary: string
  }> {
    return post('/federation', lightCurve)
  },

  // ── Catalog ─────────────────────────────────────────────────────────────

  crossMatch(ra: number, dec: number, radius_arcsec = 10): Promise<CatalogMatch> {
    return get('/catalog/cross-match', { ra, dec, radius_arcsec })
  },

  searchVarWISE(ra: number, dec: number, radius_deg = 0.5): Promise<{
    results: Array<{ name: string; ra: number; dec: number; var_type: string; period_days?: number }>
    count: number
  }> {
    return get('/catalog/varwise/search', { ra, dec, radius_deg })
  },

  // ── Leaderboard ──────────────────────────────────────────────────────────

  getLeaderboard(): Promise<Leaderboard> {
    return get('/leaderboard')
  },

  getUserStats(userId: string): Promise<UserStats> {
    return get(`/users/${userId}/stats`)
  },
}

// ── Helpers ────────────────────────────────────────────────────────────────

export function formatRA(ra: number): string {
  const h = Math.floor(ra / 15)
  const m = Math.floor(((ra / 15) - h) * 60)
  const s = (((ra / 15) - h) * 60 - m) * 60
  return `${String(h).padStart(2, '0')}h ${String(m).padStart(2, '0')}m ${s.toFixed(1)}s`
}

export function formatDec(dec: number): string {
  const sign = dec >= 0 ? '+' : '-'
  const abs = Math.abs(dec)
  const d = Math.floor(abs)
  const m = Math.floor((abs - d) * 60)
  const s = ((abs - d) * 60 - m) * 60
  return `${sign}${String(d).padStart(2, '0')}° ${String(m).padStart(2, '0')}' ${s.toFixed(0)}"`
}

export function priorityColor(priority: number): string {
  if (priority >= 80) return '#dc2626'  // red-600
  if (priority >= 60) return '#d97706'  // amber-600
  if (priority >= 40) return '#1a73e8'  // blue
  return '#5f6368'                       // muted
}

export function priorityLabel(priority: number): string {
  if (priority >= 80) return 'HIGH'
  if (priority >= 60) return 'MEDIUM'
  if (priority >= 40) return 'LOW'
  return 'ROUTINE'
}

export function confidenceBar(confidence: number): { width: string; color: string } {
  return {
    width: `${Math.round(confidence * 100)}%`,
    color: confidence >= 0.8 ? '#16a34a' : confidence >= 0.6 ? '#d97706' : '#dc2626',
  }
}
