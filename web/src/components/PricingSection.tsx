'use client';

import { useSession } from 'next-auth/react';
import Link from 'next/link';

export default function PricingSection() {
  const { data: session, status } = useSession();

  const getHref = () => {
    if (session) return '/settings/subscription';
    return '/auth/register';
  };

  const getButtonText = (tier: 'free' | 'monthly' | 'annual') => {
    if (session) {
      return tier === 'free' ? 'Go to Dashboard' : 'Manage Plan';
    }
    return tier === 'free' ? 'Get Started' : 'Subscribe';
  };

  return (
    <section id="pricing" className="py-20 bg-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-bold text-[#202124] text-center mb-4">
          Simple Pricing
        </h2>
        <p className="text-[#5f6368] text-center mb-12">
          Start free, upgrade when you need more
        </p>

        <div className="grid md:grid-cols-3 gap-8">
          {/* Free */}
          <div className="bg-white border border-[#dadce0] rounded-xl p-6">
            <h3 className="text-lg font-semibold text-[#202124] mb-2">Free</h3>
            <p className="text-[#5f6368] text-sm mb-4">For getting started</p>
            <div className="text-3xl font-bold text-[#202124] mb-6">$0</div>
            <ul className="space-y-3 mb-6 text-sm text-[#5f6368]">
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                5 analyses per month
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Basic TinyML detection
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                CSV export
              </li>
            </ul>
            <Link
              href={session ? '/dashboard' : '/auth/register'}
              className="block text-center bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium py-2.5 rounded-lg transition-colors"
            >
              {getButtonText('free')}
            </Link>
          </div>

          {/* Monthly */}
          <div className="bg-[#1a73e8] text-white rounded-xl p-6 relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-[#202124] text-white text-xs font-medium px-3 py-1 rounded-full">
              Popular
            </div>
            <h3 className="text-lg font-semibold mb-2">Monthly</h3>
            <p className="text-blue-100 text-sm mb-4">For active users</p>
            <div className="text-3xl font-bold mb-6">$9<span className="text-lg font-normal">/mo</span></div>
            <ul className="space-y-3 mb-6 text-sm">
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                50 analyses per month
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Advanced AI models
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Priority processing
              </li>
            </ul>
            <Link
              href={getHref()}
              className="block text-center bg-white hover:bg-[#f1f3f4] text-[#1a73e8] font-medium py-2.5 rounded-lg transition-colors"
            >
              {getButtonText('monthly')}
            </Link>
          </div>

          {/* Annual */}
          <div className="bg-white border border-[#dadce0] rounded-xl p-6">
            <h3 className="text-lg font-semibold text-[#202124] mb-2">Annual</h3>
            <p className="text-[#5f6368] text-sm mb-4">Best value</p>
            <div className="text-3xl font-bold text-[#202124] mb-6">$89<span className="text-lg font-normal text-[#5f6368]">/yr</span></div>
            <ul className="space-y-3 mb-6 text-sm text-[#5f6368]">
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                Unlimited analyses
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                All AI models + API
              </li>
              <li className="flex items-center gap-2">
                <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                </svg>
                White-label reports
              </li>
            </ul>
            <Link
              href={getHref()}
              className="block text-center bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium py-2.5 rounded-lg transition-colors"
            >
              {getButtonText('annual')}
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
