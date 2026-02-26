'use client'

/**
 * VerificationUI — Peer verification interface for discovery candidates
 * 3 confirmations = verified, 2 rejections = discarded
 */

import { useState } from 'react'
import { DiscoveryCandidate, discoveryClient, VerificationResult, formatRA, formatDec } from '@/lib/discovery-client'
import { CheckCircle2, XCircle, Clock, Users, Loader2, ChevronDown, ChevronUp } from 'lucide-react'

interface VerificationUIProps {
  candidate: DiscoveryCandidate
  discoveryId: string
  currentUserId: string
  initialConfirmations?: number
  initialRejections?: number
  onComplete?: (result: VerificationResult) => void
}

const CONFIRMATIONS_REQUIRED = 3
const REJECTIONS_TO_DISCARD = 2

function ProgressDots({ count, required, color }: { count: number; required: number; color: string }) {
  return (
    <div className="flex gap-1.5">
      {Array.from({ length: required }, (_, i) => (
        <div
          key={i}
          className="w-3 h-3 rounded-full border-2 transition-all"
          style={{
            backgroundColor: i < count ? color : 'transparent',
            borderColor: color,
          }}
        />
      ))}
    </div>
  )
}

export function VerificationUI({
  candidate,
  discoveryId,
  currentUserId,
  initialConfirmations = 0,
  initialRejections = 0,
  onComplete,
}: VerificationUIProps) {
  const [confirmations, setConfirmations] = useState(initialConfirmations)
  const [rejections, setRejections] = useState(initialRejections)
  const [status, setStatus] = useState<'candidate' | 'verified' | 'rejected'>('candidate')
  const [loading, setLoading] = useState(false)
  const [voted, setVoted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showDetails, setShowDetails] = useState(false)

  const { target, consensus, catalog_match } = candidate

  async function handleVote(verdict: 'confirm' | 'reject') {
    if (voted || loading || status !== 'candidate') return
    setLoading(true)
    setError(null)
    try {
      const result = await discoveryClient.verify(discoveryId, verdict, currentUserId)
      setConfirmations(result.confirmations)
      setRejections(result.rejections)
      setStatus(result.new_status as typeof status)
      setVoted(true)
      onComplete?.(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Verification failed')
    } finally {
      setLoading(false)
    }
  }

  const isResolved = status !== 'candidate'

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-[#e5e7eb] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-[#1a73e8]" />
          <span className="font-medium text-sm text-[#202124]">Community Verification</span>
        </div>
        <StatusChip status={status} />
      </div>

      {/* Object summary */}
      <div className="px-5 py-4 bg-[#f8f9fa] border-b border-[#e5e7eb]">
        <div className="flex items-start justify-between">
          <div>
            <p className="font-medium text-[#202124] text-sm">
              {consensus.consensus_label.replace(/_/g, ' ')}
            </p>
            <p className="text-xs text-[#5f6368] font-mono mt-0.5">
              {formatRA(target.ra)} &nbsp; {formatDec(target.dec)}
            </p>
          </div>
          {!catalog_match.known && (
            <span className="text-xs font-medium px-2 py-0.5 rounded bg-[#e6f4ea] text-[#137333]">
              Previously unknown
            </span>
          )}
        </div>

        {/* Expandable classification details */}
        <button
          onClick={() => setShowDetails(d => !d)}
          className="mt-3 flex items-center gap-1 text-xs text-[#1a73e8] hover:underline"
        >
          {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          Model classification details
        </button>

        {showDetails && (
          <div className="mt-3 space-y-1.5">
            {Object.entries(candidate.classifications).map(([modelId, result]) => (
              !result.error && (
                <div key={modelId} className="flex items-center justify-between text-xs">
                  <span className="text-[#5f6368] font-mono">{modelId}</span>
                  <span className="font-medium text-[#202124]">
                    {result.label.replace(/_/g, ' ')}{' '}
                    <span className="text-[#9ca3af]">({(result.confidence * 100).toFixed(0)}%)</span>
                  </span>
                </div>
              )
            ))}
          </div>
        )}
      </div>

      {/* Verification progress */}
      <div className="px-5 py-4 border-b border-[#e5e7eb]">
        <div className="grid grid-cols-2 gap-6">
          <div>
            <p className="text-xs text-[#5f6368] mb-2">
              Confirmations ({confirmations}/{CONFIRMATIONS_REQUIRED})
            </p>
            <ProgressDots count={confirmations} required={CONFIRMATIONS_REQUIRED} color="#16a34a" />
          </div>
          <div>
            <p className="text-xs text-[#5f6368] mb-2">
              Rejections ({rejections}/{REJECTIONS_TO_DISCARD})
            </p>
            <ProgressDots count={rejections} required={REJECTIONS_TO_DISCARD} color="#dc2626" />
          </div>
        </div>

        <p className="text-xs text-[#9ca3af] mt-3">
          <Clock className="w-3 h-3 inline mr-1" />
          {CONFIRMATIONS_REQUIRED} confirmations to verify • {REJECTIONS_TO_DISCARD} rejections to discard
        </p>
      </div>

      {/* Vote buttons */}
      <div className="px-5 py-4">
        {isResolved ? (
          <div className={`text-center py-3 rounded-xl text-sm font-medium ${
            status === 'verified'
              ? 'bg-[#e6f4ea] text-[#137333]'
              : 'bg-[#fce8e6] text-[#c5221f]'
          }`}>
            {status === 'verified' ? (
              <><CheckCircle2 className="w-4 h-4 inline mr-2" />Discovery Verified!</>
            ) : (
              <><XCircle className="w-4 h-4 inline mr-2" />Discovery Rejected</>
            )}
          </div>
        ) : voted ? (
          <div className="text-center py-3 rounded-xl bg-[#e8f0fe] text-[#1a73e8] text-sm">
            Vote recorded — waiting for more reviewers
          </div>
        ) : (
          <div className="space-y-3">
            <p className="text-xs text-[#5f6368] text-center">
              Review the data above. Does this look like a real discovery?
            </p>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => handleVote('confirm')}
                disabled={loading}
                className="flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium bg-[#e6f4ea] text-[#137333] hover:bg-[#ceead6] transition-colors disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
                Confirm
              </button>
              <button
                onClick={() => handleVote('reject')}
                disabled={loading}
                className="flex items-center justify-center gap-2 py-2.5 rounded-xl text-sm font-medium bg-[#fce8e6] text-[#c5221f] hover:bg-[#fad2cf] transition-colors disabled:opacity-50"
              >
                {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <XCircle className="w-4 h-4" />}
                Reject
              </button>
            </div>
          </div>
        )}

        {error && (
          <p className="mt-3 text-xs text-[#dc2626] text-center">{error}</p>
        )}
      </div>
    </div>
  )
}

function StatusChip({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    candidate: { label: 'Pending',  cls: 'bg-[#fef7e0] text-[#b45309]' },
    verified:  { label: 'Verified', cls: 'bg-[#e6f4ea] text-[#137333]' },
    rejected:  { label: 'Rejected', cls: 'bg-[#fce8e6] text-[#c5221f]' },
  }
  const cfg = map[status] ?? { label: status, cls: 'bg-[#f1f3f4] text-[#5f6368]' }
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${cfg.cls}`}>
      {cfg.label}
    </span>
  )
}
