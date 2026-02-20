'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { getCurrentUser, supabase } from '@/lib/supabase';
import { Loader2 } from 'lucide-react';

const PLANS = {
  free: {
    name: 'Free',
    price: 0,
    period: 'month',
    analyses: 5,
    features: ['5 analyses per month', 'Basic TinyML detection', 'CSV export'],
  },
  monthly: {
    name: 'Monthly',
    price: 9,
    period: 'month',
    analyses: 50,
    features: ['50 analyses per month', 'Advanced AI models', 'Priority processing', 'Email support'],
  },
  annual: {
    name: 'Annual',
    price: 89,
    period: 'year',
    analyses: -1,
    savings: 19,
    features: ['Unlimited analyses', 'All AI models + API', 'White-label reports', 'Priority support', '2 months free'],
  },
};

type PlanKey = keyof typeof PLANS;

interface UserData {
  subscription_tier: string;
  analyses_this_month: number;
  analyses_limit: number;
}

export default function SubscriptionPage() {
  const [userData, setUserData] = useState<UserData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    const { user } = await getCurrentUser();
    if (!user) {
      window.location.href = '/cloud/auth/login?redirect=/settings/subscription';
      return;
    }

    const { data } = await supabase
      .from('users')
      .select('subscription_tier, analyses_this_month, analyses_limit')
      .eq('id', user.id)
      .single();

    setUserData(data as UserData);
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin text-gray-400" />
      </div>
    );
  }

  const tier = (userData?.subscription_tier || 'free') as PlanKey;
  const currentPlan = PLANS[tier] || PLANS.free;
  const analysesUsed = userData?.analyses_this_month ?? 0;
  const analysesLimit = userData?.analyses_limit ?? 5;

  return (
    <div className="min-h-screen bg-white">
      <main className="pt-24 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="mb-8">
          <Link href="/cloud/dashboard" className="text-[#1a73e8] hover:underline text-sm mb-2 inline-flex items-center gap-1">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Dashboard
          </Link>
          <h1 className="text-3xl font-bold text-[#202124]">Usage & Billing</h1>
          <p className="text-[#5f6368]">Manage your Larun subscription</p>
        </div>

        {/* Current Plan */}
        <div className="bg-white border border-[#dadce0] rounded-xl p-6 mb-8">
          <h2 className="text-xl font-semibold text-[#202124] mb-4">Current Plan</h2>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <p className="text-2xl font-bold text-[#202124]">{currentPlan.name}</p>
              <p className="text-[#5f6368]">
                {tier === 'free' ? 'Free forever' : 'Active subscription'}
              </p>
            </div>
            {tier !== 'free' && (
              <a
                href="https://app.lemonsqueezy.com/my-orders"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm"
              >
                Manage Billing
              </a>
            )}
          </div>

          {/* Usage */}
          <div className="mt-6 pt-6 border-t border-[#dadce0]">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-[#5f6368]">Analyses used this month</span>
              <span className="text-[#202124] font-medium">
                {analysesUsed} / {analysesLimit === -1 ? '∞' : analysesLimit}
              </span>
            </div>
            {analysesLimit !== -1 && (
              <>
                <div className="bg-[#f1f3f4] rounded-full h-2">
                  <div
                    className="bg-[#1a73e8] rounded-full h-2 transition-all"
                    style={{ width: `${Math.min((analysesUsed / analysesLimit) * 100, 100)}%` }}
                  />
                </div>
                <p className="text-[#5f6368] text-sm mt-2">
                  {analysesLimit - analysesUsed} analyses remaining
                </p>
              </>
            )}
          </div>
        </div>

        {/* Upgrade Options */}
        <h2 className="text-xl font-semibold text-[#202124] mb-4">
          {tier === 'free' ? 'Upgrade Your Plan' : 'Change Plan'}
        </h2>
        <div className="grid md:grid-cols-3 gap-6 mb-10">
          {/* Free */}
          <div className={`bg-white border rounded-xl p-6 ${tier === 'free' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {tier === 'free' && <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Free</h3>
            <p className="text-[#5f6368] text-sm mb-4">For getting started</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$0</p>
            <ul className="space-y-2 mb-6">
              {PLANS.free.features.map((f, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            <div className="text-center text-[#5f6368] text-sm py-2.5">
              {tier === 'free' ? 'Current plan' : 'Basic tier'}
            </div>
          </div>

          {/* Monthly */}
          <div className={`bg-white border rounded-xl p-6 relative ${tier === 'monthly' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {tier !== 'monthly' && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-[#202124] text-white text-xs font-medium px-3 py-1 rounded-full">Popular</div>
            )}
            {tier === 'monthly' && <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Monthly</h3>
            <p className="text-[#5f6368] text-sm mb-4">For active users</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$9<span className="text-lg font-normal text-[#5f6368]">/mo</span></p>
            <ul className="space-y-2 mb-6">
              {PLANS.monthly.features.map((f, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            {tier === 'monthly' ? (
              <div className="text-center text-[#5f6368] text-sm py-2.5">Current plan</div>
            ) : (
              <a
                href="https://larunspace.lemonsqueezy.com/checkout/buy/f35b9320-79ed-462d-bdf7-1ec4841eadbb"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full text-center px-4 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm"
              >
                Upgrade
              </a>
            )}
          </div>

          {/* Annual */}
          <div className={`bg-white border rounded-xl p-6 ${tier === 'annual' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {tier !== 'annual' && <div className="text-xs font-medium text-[#34a853] mb-2">SAVE $19/YEAR</div>}
            {tier === 'annual' && <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Annual</h3>
            <p className="text-[#5f6368] text-sm mb-4">Best value</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$89<span className="text-lg font-normal text-[#5f6368]">/yr</span></p>
            <ul className="space-y-2 mb-6">
              {PLANS.annual.features.map((f, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {f}
                </li>
              ))}
            </ul>
            {tier === 'annual' ? (
              <div className="text-center text-[#5f6368] text-sm py-2.5">Current plan</div>
            ) : (
              <a
                href="https://larunspace.lemonsqueezy.com/checkout/buy/ff35095c-0eac-427c-8309-8d55448979a2"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full text-center px-4 py-2.5 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm"
              >
                Upgrade
              </a>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
