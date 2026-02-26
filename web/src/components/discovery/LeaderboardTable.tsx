'use client'

/**
 * LeaderboardTable — Ranked list of top discoverers
 */

import { Leaderboard, LeaderboardEntry } from '@/lib/discovery-client'
import { Trophy, Medal, Award, Star, Telescope } from 'lucide-react'

interface LeaderboardTableProps {
  data: Leaderboard
  currentUserId?: string
}

const RANK_META: Record<string, { icon: typeof Trophy; color: string }> = {
  'Master Astronomer': { icon: Trophy,    color: '#f59e0b' },
  'Discoverer':        { icon: Medal,     color: '#6366f1' },
  'Explorer':          { icon: Award,     color: '#3b82f6' },
  'Observer':          { icon: Star,      color: '#10b981' },
  'Stargazer':         { icon: Telescope, color: '#9ca3af' },
}

function RankIcon({ title }: { title: string }) {
  const meta = RANK_META[title]
  if (!meta) return <Telescope className="w-4 h-4 text-[#9ca3af]" />
  const Icon = meta.icon
  return <Icon className="w-4 h-4" style={{ color: meta.color }} />
}

function TopMedal({ position }: { position: number }) {
  if (position === 1) return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-b from-[#fde68a] to-[#f59e0b] flex items-center justify-center shrink-0">
      <span className="text-xs font-bold text-white">1</span>
    </div>
  )
  if (position === 2) return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-b from-[#e2e8f0] to-[#94a3b8] flex items-center justify-center shrink-0">
      <span className="text-xs font-bold text-white">2</span>
    </div>
  )
  if (position === 3) return (
    <div className="w-7 h-7 rounded-full bg-gradient-to-b from-[#fed7aa] to-[#ea580c] flex items-center justify-center shrink-0">
      <span className="text-xs font-bold text-white">3</span>
    </div>
  )
  return (
    <div className="w-7 h-7 rounded-full bg-[#f1f3f4] flex items-center justify-center shrink-0">
      <span className="text-xs text-[#5f6368]">{position}</span>
    </div>
  )
}

function EntryRow({ entry, isCurrentUser }: { entry: LeaderboardEntry; isCurrentUser: boolean }) {
  const isTop3 = entry.rank <= 3
  return (
    <div
      className={`flex items-center gap-3 px-5 py-3.5 transition-colors ${
        isCurrentUser
          ? 'bg-[#e8f0fe] border-l-2 border-[#1a73e8]'
          : isTop3
          ? 'bg-[#fafafa]'
          : 'hover:bg-[#fafafa]'
      }`}
    >
      <TopMedal position={entry.rank} />

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-[#202124] truncate">
            {entry.user_id}
            {isCurrentUser && <span className="ml-1 text-xs text-[#1a73e8]">(you)</span>}
          </span>
          <RankIcon title={entry.title} />
          <span className="text-xs text-[#9ca3af]">{entry.title}</span>
        </div>
        {entry.last_discovery && (
          <p className="text-xs text-[#9ca3af] mt-0.5 truncate">
            Last: {new Date(entry.last_discovery).toLocaleDateString()}
          </p>
        )}
      </div>

      <div className="flex items-center gap-6 shrink-0">
        <div className="text-right">
          <p className="text-sm font-semibold text-[#202124]">{entry.verified_discoveries}</p>
          <p className="text-xs text-[#9ca3af]">verified</p>
        </div>
        <div className="text-right">
          <p className="text-sm font-semibold text-[#1a73e8]">{(entry.points ?? 0).toLocaleString()}</p>
          <p className="text-xs text-[#9ca3af]">pts</p>
        </div>
      </div>
    </div>
  )
}

export function LeaderboardTable({ data, currentUserId }: LeaderboardTableProps) {
  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-[#e5e7eb]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Trophy className="w-4 h-4 text-[#f59e0b]" />
            <h3 className="font-medium text-[#202124] text-sm">Discovery Leaderboard</h3>
          </div>
          <div className="flex gap-4 text-xs text-[#5f6368]">
            <span>{(data.total_users ?? 0).toLocaleString()} explorers</span>
            <span>{(data.total_discoveries ?? 0).toLocaleString()} discoveries</span>
          </div>
        </div>
      </div>

      {/* Rankings */}
      <div className="divide-y divide-[#f1f3f4]">
        {data.rankings.length === 0 ? (
          <div className="px-5 py-12 text-center">
            <Telescope className="w-8 h-8 text-[#dadce0] mx-auto mb-3" />
            <p className="text-sm text-[#5f6368]">No discoveries yet. Be the first!</p>
          </div>
        ) : (
          data.rankings.map(entry => (
            <EntryRow
              key={entry.user_id}
              entry={entry}
              isCurrentUser={entry.user_id === currentUserId}
            />
          ))
        )}
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-[#e5e7eb] bg-[#f8f9fa]">
        <p className="text-xs text-[#9ca3af] text-center">
          Rankings update after each verified discovery • {data.period}
        </p>
      </div>
    </div>
  )
}

// ── Rank progression legend ──────────────────────────────────────────────────

export function RankLegend() {
  const ranks = [
    { title: 'Stargazer',         min: 0,  description: 'Just getting started' },
    { title: 'Observer',          min: 1,  description: '1+ verified discovery' },
    { title: 'Explorer',          min: 5,  description: '5+ verified discoveries' },
    { title: 'Discoverer',        min: 10, description: '10+ verified discoveries' },
    { title: 'Master Astronomer', min: 50, description: '50+ verified discoveries' },
  ]

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm p-5">
      <h3 className="font-medium text-[#202124] text-sm mb-4">Astronomer Ranks</h3>
      <div className="space-y-3">
        {ranks.map(r => {
          const meta = RANK_META[r.title]
          const Icon = meta?.icon ?? Telescope
          return (
            <div key={r.title} className="flex items-center gap-3">
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center shrink-0"
                style={{ backgroundColor: `${meta?.color ?? '#9ca3af'}20` }}
              >
                <Icon className="w-4 h-4" style={{ color: meta?.color ?? '#9ca3af' }} />
              </div>
              <div>
                <p className="text-sm font-medium text-[#202124]">{r.title}</p>
                <p className="text-xs text-[#9ca3af]">{r.description}</p>
              </div>
              <span className="ml-auto text-xs text-[#5f6368]">≥{r.min}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
