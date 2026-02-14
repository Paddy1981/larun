'use client'

/**
 * Quota Indicator Component
 *
 * Shows usage quota and upgrade CTA
 */

import { UsageQuota } from '@/lib/supabase'
import { AlertTriangle, ArrowRight } from 'lucide-react'
import Link from 'next/link'

interface QuotaIndicatorProps {
  quota: UsageQuota
  className?: string
}

export function QuotaIndicator({ quota, className = '' }: QuotaIndicatorProps) {
  const usagePercent = quota.quota_limit
    ? (quota.analyses_count / quota.quota_limit) * 100
    : 0

  const isNearLimit = usagePercent >= 80
  const isAtLimit = quota.quota_limit ? quota.analyses_count >= quota.quota_limit : false

  const tierName = quota.quota_limit === 100 ? 'Free' :
                   quota.quota_limit === 10000 ? 'Pro' : 'Enterprise'

  return (
    <div className={`card ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <div>
          <h4 className="text-lg font-medium">Monthly Usage</h4>
          <p className="text-sm text-larun-medium-gray">
            {tierName} Tier • Resets {new Date(new Date().getFullYear(), new Date().getMonth() + 1, 1).toLocaleDateString()}
          </p>
        </div>
        {isNearLimit && (
          <AlertTriangle className="w-6 h-6 text-yellow-500" />
        )}
      </div>

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm mb-2">
          <span className="text-larun-medium-gray">Analyses Used</span>
          <span className="font-medium text-larun-black">
            {quota.analyses_count.toLocaleString()}
            {quota.quota_limit && ` / ${quota.quota_limit.toLocaleString()}`}
          </span>
        </div>
        {quota.quota_limit && (
          <div className="w-full h-3 bg-larun-lighter-gray rounded-full overflow-hidden">
            <div
              className={`h-full transition-all ${
                isAtLimit
                  ? 'bg-red-500'
                  : isNearLimit
                  ? 'bg-yellow-500'
                  : 'bg-larun-black'
              }`}
              style={{ width: `${Math.min(usagePercent, 100)}%` }}
            />
          </div>
        )}
      </div>

      {/* Upgrade CTA */}
      {isNearLimit && quota.quota_limit !== null && (
        <div className="bg-larun-lighter-gray border border-larun-light-gray rounded-lg p-4">
          <p className="text-sm text-larun-dark-gray mb-3">
            {isAtLimit
              ? 'You\'ve reached your monthly limit. Upgrade to continue analyzing.'
              : `You're at ${usagePercent.toFixed(0)}% of your monthly quota.`}
          </p>
          <Link
            href="/pricing"
            className="btn btn-primary w-full text-sm justify-center"
          >
            Upgrade Plan
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      )}
    </div>
  )
}
