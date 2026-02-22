'use client';

import { useSession } from 'next-auth/react';
import Link from 'next/link';

/** Hero CTA buttons on /cloud — swaps targets/labels when signed in */
export function CloudHeroButtons() {
  const { data: session } = useSession();

  if (session) {
    return (
      <div className="flex gap-4 justify-center mb-16">
        <Link
          href="/cloud/analyze"
          className="inline-flex items-center justify-center bg-[#202124] text-white px-8 py-4 rounded-lg hover:bg-[#3c4043] transition-colors font-medium"
        >
          Start Analyzing
        </Link>
        <Link
          href="/dashboard"
          className="inline-flex items-center justify-center bg-white text-[#202124] px-8 py-4 rounded-lg border-2 border-[#202124] hover:bg-gray-50 transition-colors font-medium"
        >
          Dashboard
        </Link>
      </div>
    );
  }

  return (
    <div className="flex gap-4 justify-center mb-16">
      <Link
        href="/cloud/auth/signup"
        className="inline-flex items-center justify-center bg-[#202124] text-white px-8 py-4 rounded-lg hover:bg-[#3c4043] transition-colors font-medium"
      >
        Start Free Trial
      </Link>
      <Link
        href="/cloud/pricing"
        className="inline-flex items-center justify-center bg-white text-[#202124] px-8 py-4 rounded-lg border-2 border-[#202124] hover:bg-gray-50 transition-colors font-medium"
      >
        View Pricing
      </Link>
    </div>
  );
}

/** Bottom "Get Started" CTA on /cloud */
export function CloudBottomCTA() {
  const { data: session } = useSession();

  return (
    <Link
      href={session ? '/cloud/analyze' : '/cloud/auth/signup'}
      className="inline-flex items-center justify-center bg-[#202124] text-white px-8 py-4 rounded-lg hover:bg-[#3c4043] transition-colors font-medium"
    >
      {session ? 'Start Analyzing Now →' : 'Start Analyzing Now →'}
    </Link>
  );
}
