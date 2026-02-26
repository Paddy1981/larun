'use client'

/**
 * DiscoveryReport — Shareable full-page report for a discovery run
 */

import { DiscoveryReport as Report, DiscoveryCandidate, formatRA, formatDec, priorityColor, priorityLabel } from '@/lib/discovery-client'
import { Download, Share2, CheckCircle2, AlertTriangle, Star, BarChart2 } from 'lucide-react'
import { useState } from 'react'

interface DiscoveryReportProps {
  report: Report
  onDownload?: () => void
}

function SummaryBadge({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div className="bg-[#f8f9fa] rounded-xl px-4 py-3 text-center">
      <p className="text-2xl font-semibold" style={{ color: accent ?? '#202124' }}>{value}</p>
      <p className="text-xs text-[#5f6368] mt-0.5">{label}</p>
    </div>
  )
}

function CandidateRow({ c, idx }: { c: DiscoveryCandidate; idx: number }) {
  const pColor = priorityColor(c.priority)
  return (
    <tr className="border-b border-[#f1f3f4] hover:bg-[#fafafa] transition-colors">
      <td className="py-3 px-4 text-sm text-[#5f6368]">{idx + 1}</td>
      <td className="py-3 px-4">
        <p className="text-sm font-medium text-[#202124]">
          {c.consensus.consensus_label.replace(/_/g, ' ')}
        </p>
        <p className="text-xs text-[#9ca3af] font-mono">
          {formatRA(c.target.ra)} / {formatDec(c.target.dec)}
        </p>
      </td>
      <td className="py-3 px-4">
        <span
          className="text-xs font-medium px-2 py-0.5 rounded-full"
          style={{ color: pColor, backgroundColor: `${pColor}15`, border: `1px solid ${pColor}35` }}
        >
          {priorityLabel(c.priority)} ({c.priority})
        </span>
      </td>
      <td className="py-3 px-4 text-sm text-[#202124]">
        {(c.consensus.consensus_confidence * 100).toFixed(0)}%
      </td>
      <td className="py-3 px-4 text-sm text-[#202124]">
        {(c.novelty_score * 100).toFixed(0)}%
      </td>
      <td className="py-3 px-4">
        <div className="flex gap-1">
          {c.consensus.is_variable && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-[#e8f0fe] text-[#1a73e8]">VAR</span>
          )}
          {c.consensus.anomaly_detected && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-[#fce8e6] text-[#c5221f]">ANO</span>
          )}
          {!c.catalog_match.known && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-[#e6f4ea] text-[#137333]">NEW</span>
          )}
        </div>
      </td>
    </tr>
  )
}

function ModelCoverage({ report }: { report: Report }) {
  // Collect all unique model IDs across all candidates
  const allCandidates = [...report.candidates, ...report.known, ...report.anomalies]
  const modelIds = Array.from(
    new Set(allCandidates.flatMap(c => Object.keys(c.classifications)))
  )

  if (modelIds.length === 0) return null

  // Count how many had each model run successfully
  const counts = modelIds.map(id => {
    const ran = allCandidates.filter(c => c.classifications[id] && !c.classifications[id].error).length
    return { id, ran, total: allCandidates.length }
  })

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
      <div className="flex items-center gap-2 mb-4">
        <BarChart2 className="w-4 h-4 text-[#1a73e8]" />
        <h3 className="font-medium text-[#202124] text-sm">Model Coverage</h3>
      </div>
      <div className="space-y-2">
        {counts.map(({ id, ran, total }) => (
          <div key={id} className="flex items-center gap-3">
            <span className="text-xs font-mono text-[#5f6368] w-36 shrink-0">{id}</span>
            <div className="flex-1 bg-[#f1f3f4] rounded-full h-1.5 overflow-hidden">
              <div
                className="h-full bg-[#1a73e8] rounded-full"
                style={{ width: total > 0 ? `${(ran / total) * 100}%` : '0%' }}
              />
            </div>
            <span className="text-xs text-[#9ca3af] w-12 text-right">{ran}/{total}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function DiscoveryReport({ report, onDownload }: DiscoveryReportProps) {
  const [copied, setCopied] = useState(false)

  function handleShare() {
    if (typeof navigator !== 'undefined' && navigator.clipboard) {
      const url = typeof window !== 'undefined' ? window.location.href : ''
      navigator.clipboard.writeText(url).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      })
    }
  }

  function handleDownload() {
    if (onDownload) {
      onDownload()
      return
    }
    const json = JSON.stringify(report, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `larun-discovery-ra${report.meta.ra.toFixed(2)}-dec${report.meta.dec.toFixed(2)}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const totalAnalyzed = report.stats.analyzed
  const newCount = report.candidates.filter(c => !c.catalog_match.known).length
  const highPriority = report.candidates.filter(c => c.priority >= 80).length

  return (
    <div className="space-y-5">
      {/* Report header */}
      <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-lg font-semibold text-[#202124] mb-1">Discovery Report</h2>
            <p className="text-sm text-[#5f6368]">
              RA {formatRA(report.meta.ra)} &nbsp;/&nbsp; Dec {formatDec(report.meta.dec)} &nbsp;·&nbsp;
              r = {report.meta.radius_deg}° &nbsp;·&nbsp;
              Sources: {report.meta.sources.join(', ').toUpperCase()}
            </p>
            <p className="text-xs text-[#9ca3af] mt-1">
              Analyzed in {report.stats.elapsed_seconds.toFixed(1)}s &nbsp;·&nbsp; Models: {report.meta.models_used}
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={handleShare}
              className="flex items-center gap-1.5 btn btn-outline text-sm py-2 px-3"
            >
              <Share2 className="w-3.5 h-3.5" />
              {copied ? 'Copied!' : 'Share'}
            </button>
            <button
              onClick={handleDownload}
              className="flex items-center gap-1.5 btn btn-primary text-sm py-2 px-3"
            >
              <Download className="w-3.5 h-3.5" />
              JSON
            </button>
          </div>
        </div>

        {/* Summary stats */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-5">
          <SummaryBadge label="Analyzed" value={totalAnalyzed} />
          <SummaryBadge label="Candidates" value={report.stats.candidates} accent="#1a73e8" />
          <SummaryBadge label="Previously Unknown" value={newCount} accent="#137333" />
          <SummaryBadge label="High Priority" value={highPriority} accent="#dc2626" />
        </div>
      </div>

      {/* Discovery candidates table */}
      {report.candidates.length > 0 && (
        <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-[#e5e7eb] flex items-center gap-2">
            <Star className="w-4 h-4 text-[#f59e0b]" />
            <h3 className="font-medium text-[#202124] text-sm">
              Discovery Candidates ({report.candidates.length})
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-[#f8f9fa] text-xs text-[#5f6368]">
                  <th className="py-2.5 px-4 text-left font-medium">#</th>
                  <th className="py-2.5 px-4 text-left font-medium">Classification / Coords</th>
                  <th className="py-2.5 px-4 text-left font-medium">Priority</th>
                  <th className="py-2.5 px-4 text-left font-medium">Confidence</th>
                  <th className="py-2.5 px-4 text-left font-medium">Novelty</th>
                  <th className="py-2.5 px-4 text-left font-medium">Flags</th>
                </tr>
              </thead>
              <tbody>
                {report.candidates.map((c, i) => (
                  <CandidateRow key={i} c={c} idx={i} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Anomalies section */}
      {report.anomalies.length > 0 && (
        <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-[#e5e7eb] flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-[#dc2626]" />
            <h3 className="font-medium text-[#202124] text-sm">
              Anomalous Objects ({report.anomalies.length})
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-[#f8f9fa] text-xs text-[#5f6368]">
                  <th className="py-2.5 px-4 text-left font-medium">#</th>
                  <th className="py-2.5 px-4 text-left font-medium">Classification / Coords</th>
                  <th className="py-2.5 px-4 text-left font-medium">Priority</th>
                  <th className="py-2.5 px-4 text-left font-medium">Confidence</th>
                  <th className="py-2.5 px-4 text-left font-medium">Novelty</th>
                  <th className="py-2.5 px-4 text-left font-medium">Flags</th>
                </tr>
              </thead>
              <tbody>
                {report.anomalies.map((c, i) => (
                  <CandidateRow key={i} c={c} idx={i} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Known objects summary */}
      {report.known.length > 0 && (
        <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
          <div className="flex items-center gap-2 mb-3">
            <CheckCircle2 className="w-4 h-4 text-[#16a34a]" />
            <h3 className="font-medium text-[#202124] text-sm">
              Known Objects ({report.known.length})
            </h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {report.known.slice(0, 20).map((c, i) => (
              <span key={i} className="text-xs px-2 py-1 rounded-full bg-[#f1f3f4] text-[#5f6368]">
                {c.catalog_match.matches?.[0]?.name ??
                  `${c.target.ra.toFixed(2)},${c.target.dec.toFixed(2)}`}
              </span>
            ))}
            {report.known.length > 20 && (
              <span className="text-xs px-2 py-1 text-[#9ca3af]">+{report.known.length - 20} more</span>
            )}
          </div>
        </div>
      )}

      {/* Model coverage */}
      <ModelCoverage report={report} />
    </div>
  )
}
