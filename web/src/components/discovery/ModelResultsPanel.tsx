'use client'

/**
 * ModelResultsPanel — Displays all Layer-2 model outputs from the Federation
 */

import { ClassificationResult, ConsensusResult, confidenceBar } from '@/lib/discovery-client'
import { CheckCircle2, AlertTriangle, Minus, Clock } from 'lucide-react'

interface ModelResultsPanelProps {
  results: Record<string, ClassificationResult>
  consensus?: ConsensusResult
}

const MODEL_LABELS: Record<string, { name: string; color: string }> = {
  'VARDET-001':     { name: 'VARnet Variability',  color: '#1a73e8' },
  'ANOMALY-001':    { name: 'Anomaly Detector',    color: '#dc2626' },
  'DEBLEND-001':    { name: 'Blend Detector',      color: '#d97706' },
  'PERIODOGRAM-001':{ name: 'Period Finder',       color: '#7c3aed' },
  'EXOPLANET-001':  { name: 'Transit Detector',    color: '#0891b2' },
  'VSTAR-001':      { name: 'Variable Star Class', color: '#059669' },
  'FLARE-001':      { name: 'Flare Detector',      color: '#ea580c' },
  'ASTERO-001':     { name: 'Asteroseismology',    color: '#6d28d9' },
  'SUPERNOVA-001':  { name: 'Transient Detector',  color: '#be185d' },
  'MICROLENS-001':  { name: 'Microlensing',        color: '#0f766e' },
  'GALAXY-001':     { name: 'Galaxy Morphology',   color: '#1d4ed8' },
  'SPECTYPE-001':   { name: 'Spectral Type',       color: '#374151' },
}

function labelBadge(label: string, modelId: string) {
  const color = MODEL_LABELS[modelId]?.color ?? '#5f6368'
  const isNegative = ['NON_VARIABLE', 'NORMAL', 'CLEAN', 'NO_PERIOD', 'no_flare', 'noise', 'non_pulsating', 'no_transient', 'no_event', 'constant'].includes(label.toLowerCase())
  const bgColor = isNegative ? '#f1f3f4' : `${color}18`
  const textColor = isNegative ? '#5f6368' : color
  const borderColor = isNegative ? '#e5e7eb' : `${color}40`

  return (
    <span
      className="text-xs font-medium px-2 py-0.5 rounded-full border"
      style={{ backgroundColor: bgColor, color: textColor, borderColor }}
    >
      {label.replace(/_/g, ' ')}
    </span>
  )
}

function ConsensusBar({ label, value, total }: { label: string; value: number; total: number }) {
  const pct = total > 0 ? (value / total) * 100 : 0
  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-[#5f6368] w-28 shrink-0">{label}</span>
      <div className="flex-1 bg-[#f1f3f4] rounded-full h-1.5 overflow-hidden">
        <div className="h-full bg-[#1a73e8] rounded-full transition-all" style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-[#202124] w-8 text-right">{value}</span>
    </div>
  )
}

export function ModelResultsPanel({ results, consensus }: ModelResultsPanelProps) {
  const entries = Object.entries(results)
  if (entries.length === 0) {
    return (
      <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-6 text-center">
        <span className="text-sm text-[#5f6368]">No model results yet</span>
      </div>
    )
  }

  const successEntries = entries.filter(([, r]) => !r.error)
  const errorEntries = entries.filter(([, r]) => !!r.error)

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-[#e5e7eb] flex items-center justify-between">
        <h3 className="font-medium text-[#202124] text-sm">Federation Results</h3>
        <span className="text-xs text-[#5f6368]">
          {successEntries.length}/{entries.length} models
        </span>
      </div>

      {/* Consensus banner */}
      {consensus && (
        <div className="px-5 py-3 bg-[#f8f9fa] border-b border-[#e5e7eb]">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-xs text-[#5f6368] mb-1">Consensus</p>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-base font-semibold text-[#202124]">
                  {consensus.consensus_label.replace(/_/g, ' ')}
                </span>
                <span className="text-sm text-[#5f6368]">
                  {(consensus.consensus_confidence * 100).toFixed(0)}% confidence
                </span>
              </div>
              <div className="flex gap-3 mt-2">
                {consensus.is_variable && (
                  <span className="flex items-center gap-1 text-xs text-[#1a73e8]">
                    <CheckCircle2 className="w-3 h-3" /> Variable
                  </span>
                )}
                {consensus.anomaly_detected && (
                  <span className="flex items-center gap-1 text-xs text-[#dc2626]">
                    <AlertTriangle className="w-3 h-3" /> Anomaly
                  </span>
                )}
                {consensus.blend_detected && (
                  <span className="flex items-center gap-1 text-xs text-[#d97706]">
                    <AlertTriangle className="w-3 h-3" /> Blend
                  </span>
                )}
                {!consensus.is_variable && !consensus.anomaly_detected && !consensus.blend_detected && (
                  <span className="flex items-center gap-1 text-xs text-[#5f6368]">
                    <Minus className="w-3 h-3" /> No signals detected
                  </span>
                )}
              </div>
            </div>
            <div className="text-right shrink-0">
              <p className="text-xs text-[#5f6368] mb-1">Agreement</p>
              <p className="text-lg font-semibold text-[#202124]">{consensus.agreement_count}</p>
              <p className="text-xs text-[#5f6368]">models</p>
            </div>
          </div>
        </div>
      )}

      {/* Per-model rows */}
      <div className="divide-y divide-[#f1f3f4]">
        {successEntries.map(([modelId, result]) => {
          const meta = MODEL_LABELS[modelId]
          const bar = confidenceBar(result.confidence)
          const hasInferenceTime = result.inference_ms !== undefined && result.inference_ms >= 0
          return (
            <div key={modelId} className="px-5 py-3 hover:bg-[#fafafa] transition-colors">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div
                    className="w-2 h-2 rounded-full shrink-0"
                    style={{ backgroundColor: meta?.color ?? '#5f6368' }}
                  />
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-[#202124] truncate">
                      {meta?.name ?? modelId}
                    </p>
                    <p className="text-xs text-[#9ca3af] font-mono">{modelId}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  {labelBadge(result.label, modelId)}
                  <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-1.5">
                      <div className="w-16 h-1.5 bg-[#f1f3f4] rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: bar.width, backgroundColor: bar.color }}
                        />
                      </div>
                      <span className="text-xs text-[#5f6368] w-8 text-right">
                        {(result.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    {hasInferenceTime && (
                      <span className="flex items-center gap-1 text-[10px] text-[#9ca3af]">
                        <Clock className="w-2.5 h-2.5" />
                        {result.inference_ms!.toFixed(1)} ms
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {/* Probability breakdown (if available) */}
              {result.probabilities && Object.keys(result.probabilities).length > 1 && (
                <div className="mt-2 space-y-1">
                  {Object.entries(result.probabilities)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 3)
                    .map(([cls, prob]) => (
                      <ConsensusBar
                        key={cls}
                        label={cls.replace(/_/g, ' ')}
                        value={+(prob * 100).toFixed(0)}
                        total={100}
                      />
                    ))}
                </div>
              )}
            </div>
          )
        })}

        {/* Errored models */}
        {errorEntries.map(([modelId, result]) => (
          <div key={modelId} className="px-5 py-3 opacity-50">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-[#9ca3af]" />
              <span className="text-xs text-[#5f6368]">
                {MODEL_LABELS[modelId]?.name ?? modelId} — {result.error}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
