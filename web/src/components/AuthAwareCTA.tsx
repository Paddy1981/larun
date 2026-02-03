'use client';

import { useSession } from 'next-auth/react';
import Link from 'next/link';

interface AuthAwareCTAProps {
  className?: string;
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'outline';
}

// For logged-in users, go to subscription settings
// For non-logged-in users, go to register
export function AuthAwareCTA({ className = '', children, variant = 'primary' }: AuthAwareCTAProps) {
  const { data: session, status } = useSession();

  const href = session ? '/settings/subscription' : '/auth/register';

  if (status === 'loading') {
    return (
      <span className={className}>
        {children}
      </span>
    );
  }

  return (
    <Link href={href} className={className}>
      {children}
    </Link>
  );
}

// Pricing card button that's session-aware
export function PricingButton({
  tier,
  className = ''
}: {
  tier: 'free' | 'monthly' | 'annual';
  className?: string;
}) {
  const { data: session, status } = useSession();

  // For logged-in users, go to subscription page
  // For non-logged-in users, go to register
  const href = session ? '/settings/subscription' : '/auth/register';

  const labels = {
    free: 'Get Started',
    monthly: 'Subscribe',
    annual: 'Subscribe'
  };

  if (status === 'loading') {
    return (
      <span className={`block text-center font-medium py-2.5 rounded-lg ${className}`}>
        {labels[tier]}
      </span>
    );
  }

  return (
    <Link href={href} className={`block text-center font-medium py-2.5 rounded-lg transition-colors ${className}`}>
      {session ? (tier === 'free' ? 'Go to Dashboard' : 'Manage Subscription') : labels[tier]}
    </Link>
  );
}
