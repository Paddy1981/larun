'use client'

/**
 * CandidateCard — Single discovery candidate summary card
 */

import { DiscoveryCandidate, formatRA, formatDec, priorityColor, priorityLabel } from '@/lib/discovery-client'
import { Star, Zap, Shuffle, AlertCircle, ChevronRight, Database } from 'lucide-react'

interface CandidateCardProps {
  candidate: DiscoveryCandidate
  onVerify?: (candidate: DiscoveryCandidate) => void
  onViewDetails?: (candidate: DiscoveryCandidate) => void
  rank?: number
}

function SourceBadge({ source }: { source: string }) {
  const colors: Record<string, string> = {
    tess:    'bg-[#e8f0fe] text-[#1a73e8]',
    kepler:  'bg-[#fce8e6] text-[#c5221f]',
    neowise: 'bg-[#fef7e0] text-[#b45309]',
  }
  const cls = colors[source.toLowerCase()] ?? 'bg-[#f1f3f4] text-[#5f6368]'
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${cls}`}>
      {source.toUpperCase()}
    </span>
  )
}

function ClassIcon({ label }: { label: string }) {
  const l = label.toLowerCase()
  if (l.includes('pulsator') || l.includes('cepheid') || l.includes('rr_lyrae')) {
    return <Star className="w-4 h-4 text-[#7c3aed]" />
  }
  if (l.includes('transient') || l.includes('flare') || l.includes('supernova')) {
    return <Zap className="w-4 h-4 text-[#dc2626]" />
  }
  if (l.includes('eclipsing') || l.includes('blend')) {
    return <Shuffle className="w-4 h-4 text-[#d97706]" />
  }
  if (l.includes('anomaly') || l.includes('unknown')) {
    return <AlertCircle className="w-4 h-4 text-[#dc2626]" />
  }
  return <Database className="w-4 h-4 text-[#5f6368]" />
}

export function CandidateCard({ candidate, onVerify, onViewDetails, rank }: CandidateCardProps) {
  const { target, consensus, catalog_match, priority, source, novelty_score, period_days, light_curve_meta } = candidate

  const pColor = priorityColor(priority)
  const pLabel = priorityLabel(priority)
  const isNew = !catalog_match.known

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm hover:shadow-md transition-shadow p-5">
      {/* Top row */}
      <div className="flex items-start justify-between gap-3 mb-4">
        <div className="flex items-center gap-3">
          {rank !== undefined && (
            <span className="w-7 h-7 rounded-full bg-[#f1f3f4] text-[#5f6368] text-xs font-medium flex items-center justify-center shrink-0">
              {rank}
            </span>
          )}
          <div>
            <div className="flex items-center gap-2">
              <ClassIcon label={consensus.consensus_label} />
              <span className="font-medium text-[#202124] text-sm">
                {consensus.consensus_label.replace(/_/g, ' ')}
              </span>
              {isNew && (
                <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-[#e6f4ea] text-[#137333] border border-[#ceead6]">
                  NEW
                </span>
              )}
            </div>
            <p className="text-xs text-[#5f6368] font-mono mt-0.5">
              {formatRA(target.ra)} &nbsp; {formatDec(target.dec)}
            </p>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span
            className="text-xs font-semibold px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: `${pColor}18`,
              color: pColor,
              border: `1px solid ${pColor}40`,
            }}
          >
            {pLabel}
          </span>
          <span className="text-xs text-[#5f6368]">Priority {priority}</span>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <Stat label="Confidence" value={`${(consensus.consensus_confidence * 100).toFixed(0)}%`} />
        <Stat label="Novelty" value={`${(novelty_score * 100).toFixed(0)}%`} />
        <Stat label="Data points" value={String(light_curve_meta.n_points)} />
        {period_days !== undefined && period_days > 0
          ? <Stat label="Period" value={`${period_days.toFixed(3)} d`} />
          : <Stat label="Source" value={<SourceBadge source={source} />} />
        }
      </div>

      {/* Flags */}
      <div className="flex flex-wrap gap-2 mb-4">
        {consensus.is_variable && (
          <Flag color="#1a73e8" label="Variable" />
        )}
        {consensus.anomaly_detected && (
          <Flag color="#dc2626" label="Anomaly" />
        )}
        {consensus.blend_detected && (
          <Flag color="#d97706" label="Blended" />
        )}
        {catalog_match.matches?.length > 0 && (
          <Flag color="#5f6368" label={`Matched: ${catalog_match.matches[0].catalog}`} />
        )}
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        {onViewDetails && (
          <button
            onClick={() => onViewDetails(candidate)}
            className="flex-1 flex items-center justify-center gap-1.5 btn btn-outline text-sm py-2"
          >
            Details
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        )}
        {onVerify && isNew && (
          <button
            onClick={() => onVerify(candidate)}
            className="flex-1 btn btn-primary text-sm py-2"
          >
            Verify
          </button>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | React.ReactNode }) {
  return (
    <div className="bg-[#f8f9fa] rounded-lg px-3 py-2">
      <p className="text-xs text-[#5f6368] mb-0.5">{label}</p>
      <p className="text-sm font-medium text-[#202124]">{value}</p>
    </div>
  )
}

function Flag({ color, label }: { color: string; label: string }) {
  return (
    <span
      className="text-xs px-2 py-0.5 rounded-full border"
      style={{ color, backgroundColor: `${color}12`, borderColor: `${color}30` }}
    >
      {label}
    </span>
  )
}
