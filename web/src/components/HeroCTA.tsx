'use client';

import { useSession } from 'next-auth/react';
import Link from 'next/link';

export function HeroButtons() {
  const { data: session } = useSession();

  return (
    <div className="flex flex-col sm:flex-row gap-4 justify-center">
      <Link
        href={session ? '/dashboard' : '/auth/register'}
        className="bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium px-8 py-3.5 rounded-lg transition-colors text-base"
      >
        {session ? 'Go to Dashboard' : 'Start Exploring - Free'}
      </Link>
      <Link
        href="/dashboard"
        className="bg-white hover:bg-[#f1f3f4] text-[#202124] font-medium px-8 py-3.5 rounded-lg border border-[#dadce0] transition-colors text-base"
      >
        View Demo
      </Link>
    </div>
  );
}

export function BottomCTA() {
  const { data: session } = useSession();

  return (
    <section className="py-20 bg-[#202124] text-white">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <h2 className="text-3xl md:text-4xl font-bold mb-6">
          Start Exploring the Cosmos
        </h2>
        <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
          Join researchers and astronomy enthusiasts discovering new worlds with AI-powered analysis. No credit card required.
        </p>
        <Link
          href={session ? '/dashboard' : '/auth/register'}
          className="inline-block bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium px-8 py-3.5 rounded-lg transition-colors"
        >
          {session ? 'Go to Dashboard' : 'Create Free Account'}
        </Link>
      </div>
    </section>
  );
}
