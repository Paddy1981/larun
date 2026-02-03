'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSession } from 'next-auth/react';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';

interface SubscriptionData {
  plan: 'free' | 'monthly' | 'annual';
  status: 'active' | 'canceled' | 'past_due';
  current_period_end: string;
  cancel_at_period_end: boolean;
  analyses_used: number;
  analyses_limit: number;
}

const PLANS = {
  free: {
    name: 'Free',
    price: 0,
    period: 'month',
    analyses: 3,
    features: [
      '3 analyses per month',
      'Basic TinyML detection',
      'CSV export',
    ],
  },
  monthly: {
    name: 'Monthly',
    price: 9,
    period: 'month',
    analyses: 50,
    features: [
      '50 analyses per month',
      'Advanced AI models',
      'Priority processing',
      'Email support',
    ],
  },
  annual: {
    name: 'Annual',
    price: 89,
    period: 'year',
    analyses: -1,
    savings: 19,
    features: [
      'Unlimited analyses',
      'All AI models + API',
      'White-label reports',
      'Priority support',
      '2 months free',
    ],
  },
};

export default function SubscriptionPage() {
  const { data: session, status } = useSession();
  const router = useRouter();
  const [subscription, setSubscription] = useState<SubscriptionData>({
    plan: 'free',
    status: 'active',
    current_period_end: '2026-03-02',
    cancel_at_period_end: false,
    analyses_used: 1,
    analyses_limit: 3,
  });
  const [isLoading, setIsLoading] = useState(false);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      router.push('/auth/login?callbackUrl=/settings/subscription');
    }
  }, [status, router]);

  const handleManageBilling = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/subscription/portal', {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        window.location.href = data.url;
      } else {
        alert('Billing portal not configured yet. Coming soon!');
      }
    } catch (error) {
      console.error('Failed to open billing portal:', error);
      alert('Billing portal not configured yet. Coming soon!');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpgrade = async (plan: string) => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/subscription/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plan }),
      });
      if (res.ok) {
        const data = await res.json();
        window.location.href = data.url;
      } else {
        alert('Checkout not configured yet. Coming soon!');
      }
    } catch (error) {
      console.error('Failed to create checkout session:', error);
      alert('Checkout not configured yet. Coming soon!');
    } finally {
      setIsLoading(false);
    }
  };

  // Show loading while checking auth
  if (status === 'loading') {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-[#1a73e8] border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-[#5f6368]">Loading...</p>
        </div>
      </div>
    );
  }

  // Don't render if not authenticated
  if (status === 'unauthenticated') {
    return null;
  }

  const currentPlan = PLANS[subscription.plan];

  return (
    <div className="min-h-screen bg-white">
      <Header />

      <main className="pt-24 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <div className="mb-8">
          <Link href="/dashboard" className="text-[#1a73e8] hover:underline text-sm mb-2 inline-flex items-center gap-1">
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
              <p className="text-2xl font-bold text-[#202124] capitalize">{currentPlan.name}</p>
              <p className="text-[#5f6368]">
                {subscription.status === 'active' ? (
                  subscription.plan === 'free' ? (
                    'Free forever'
                  ) : (
                    <>Renews on {new Date(subscription.current_period_end).toLocaleDateString()}</>
                  )
                ) : subscription.status === 'canceled' ? (
                  <>Expires on {new Date(subscription.current_period_end).toLocaleDateString()}</>
                ) : (
                  <span className="text-[#ea4335]">Payment past due</span>
                )}
              </p>
            </div>
            {subscription.plan !== 'free' && (
              <button
                onClick={handleManageBilling}
                disabled={isLoading}
                className="px-4 py-2 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm disabled:opacity-50"
              >
                Manage Billing
              </button>
            )}
          </div>

          {/* Usage */}
          <div className="mt-6 pt-6 border-t border-[#dadce0]">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-[#5f6368]">Analyses used this month</span>
              <span className="text-[#202124] font-medium">
                {subscription.analyses_used} / {subscription.analyses_limit === -1 ? '∞' : subscription.analyses_limit}
              </span>
            </div>
            {subscription.analyses_limit !== -1 && (
              <>
                <div className="bg-[#f1f3f4] rounded-full h-2">
                  <div
                    className="bg-[#1a73e8] rounded-full h-2 transition-all"
                    style={{ width: `${Math.min((subscription.analyses_used / subscription.analyses_limit) * 100, 100)}%` }}
                  />
                </div>
                <p className="text-[#5f6368] text-sm mt-2">
                  {subscription.analyses_limit - subscription.analyses_used} analyses remaining
                </p>
              </>
            )}
          </div>
        </div>

        {/* Upgrade Options */}
        <h2 className="text-xl font-semibold text-[#202124] mb-4">
          {subscription.plan === 'free' ? 'Upgrade Your Plan' : 'Change Plan'}
        </h2>
        <div className="grid md:grid-cols-3 gap-6 mb-10">
          {/* Free */}
          <div className={`bg-white border rounded-xl p-6 ${subscription.plan === 'free' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {subscription.plan === 'free' && (
              <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>
            )}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Free</h3>
            <p className="text-[#5f6368] text-sm mb-4">For getting started</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$0</p>
            <ul className="space-y-2 mb-6">
              {PLANS.free.features.map((feature, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {feature}
                </li>
              ))}
            </ul>
            {subscription.plan === 'free' ? (
              <div className="text-center text-[#5f6368] text-sm py-2.5">Current plan</div>
            ) : (
              <button
                onClick={() => handleUpgrade('free')}
                disabled={isLoading}
                className="w-full px-4 py-2.5 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm disabled:opacity-50"
              >
                Downgrade
              </button>
            )}
          </div>

          {/* Monthly */}
          <div className={`bg-white border rounded-xl p-6 relative ${subscription.plan === 'monthly' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {subscription.plan !== 'monthly' && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-[#202124] text-white text-xs font-medium px-3 py-1 rounded-full">
                Popular
              </div>
            )}
            {subscription.plan === 'monthly' && (
              <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>
            )}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Monthly</h3>
            <p className="text-[#5f6368] text-sm mb-4">For active users</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$9<span className="text-lg font-normal text-[#5f6368]">/mo</span></p>
            <ul className="space-y-2 mb-6">
              {PLANS.monthly.features.map((feature, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {feature}
                </li>
              ))}
            </ul>
            {subscription.plan === 'monthly' ? (
              <div className="text-center text-[#5f6368] text-sm py-2.5">Current plan</div>
            ) : (
              <button
                onClick={() => handleUpgrade('monthly')}
                disabled={isLoading}
                className="w-full px-4 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm disabled:opacity-50"
              >
                {subscription.plan === 'free' ? 'Upgrade' : 'Switch'}
              </button>
            )}
          </div>

          {/* Annual */}
          <div className={`bg-white border rounded-xl p-6 ${subscription.plan === 'annual' ? 'border-[#1a73e8] border-2' : 'border-[#dadce0]'}`}>
            {subscription.plan !== 'annual' && (
              <div className="text-xs font-medium text-[#34a853] mb-2">SAVE $19/YEAR</div>
            )}
            {subscription.plan === 'annual' && (
              <div className="text-xs font-medium text-[#1a73e8] mb-2">CURRENT PLAN</div>
            )}
            <h3 className="text-lg font-semibold text-[#202124] mb-1">Annual</h3>
            <p className="text-[#5f6368] text-sm mb-4">Best value</p>
            <p className="text-3xl font-bold text-[#202124] mb-4">$89<span className="text-lg font-normal text-[#5f6368]">/yr</span></p>
            <ul className="space-y-2 mb-6">
              {PLANS.annual.features.map((feature, i) => (
                <li key={i} className="text-[#5f6368] text-sm flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  {feature}
                </li>
              ))}
            </ul>
            {subscription.plan === 'annual' ? (
              <div className="text-center text-[#5f6368] text-sm py-2.5">Current plan</div>
            ) : (
              <button
                onClick={() => handleUpgrade('annual')}
                disabled={isLoading}
                className="w-full px-4 py-2.5 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors text-sm disabled:opacity-50"
              >
                {subscription.plan === 'free' ? 'Upgrade' : 'Switch'}
              </button>
            )}
          </div>
        </div>

        {/* FAQ */}
        <div className="bg-[#f8f9fa] rounded-xl p-6">
          <h2 className="text-xl font-semibold text-[#202124] mb-4">Frequently Asked Questions</h2>
          <div className="space-y-4">
            <div>
              <h3 className="text-[#202124] font-medium">Can I cancel anytime?</h3>
              <p className="text-[#5f6368] text-sm mt-1">
                Yes, you can cancel your subscription at any time. You'll continue to have access
                until the end of your billing period.
              </p>
            </div>
            <div>
              <h3 className="text-[#202124] font-medium">What happens to my analyses if I downgrade?</h3>
              <p className="text-[#5f6368] text-sm mt-1">
                All your previous analyses and results remain accessible. You'll just be limited
                to the lower plan's monthly analysis quota going forward.
              </p>
            </div>
            <div>
              <h3 className="text-[#202124] font-medium">Do unused analyses roll over?</h3>
              <p className="text-[#5f6368] text-sm mt-1">
                No, unused analyses do not roll over to the next month. Each billing period
                starts fresh with your plan's full allocation.
              </p>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="flex justify-center gap-6 mb-4">
            <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm hover:text-[#202124]">
              laruneng.com
            </a>
            <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm hover:text-[#202124]">
              GitHub
            </a>
            <Link href="/guide" className="text-[#5f6368] text-sm hover:text-[#202124]">
              Documentation
            </Link>
          </div>
          <p className="text-xs text-[#5f6368]">&copy; {new Date().getFullYear()} Larun Engineering. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
