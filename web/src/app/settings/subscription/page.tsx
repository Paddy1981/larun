'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button, Card } from '@/components/ui';
import { Header, Footer } from '@/components/layout';

interface SubscriptionData {
  plan: 'free' | 'hobbyist' | 'professional';
  status: 'active' | 'canceled' | 'past_due';
  current_period_end: string;
  cancel_at_period_end: boolean;
  analyses_used: number;
  analyses_limit: number;
}

const PLANS = {
  hobbyist_monthly: {
    name: 'Hobbyist Monthly',
    price: 9,
    period: 'month',
    analyses: 25,
    features: [
      '25 analyses per month',
      'Basic vetting tests',
      'Phase-folded visualizations',
      'Email support',
    ],
  },
  hobbyist_annual: {
    name: 'Hobbyist Annual',
    price: 89,
    period: 'year',
    analyses: 25,
    savings: 19,
    features: [
      '25 analyses per month',
      'Basic vetting tests',
      'Phase-folded visualizations',
      'Email support',
      '2 months free',
    ],
  },
  professional: {
    name: 'Professional',
    price: 49,
    period: 'month',
    analyses: -1,
    features: [
      'Unlimited analyses',
      'Advanced vetting suite',
      'MCMC uncertainty estimation',
      'API access',
      'Priority support',
      'Citation-ready reports',
    ],
  },
};

export default function SubscriptionPage() {
  const [subscription] = useState<SubscriptionData>({
    plan: 'hobbyist',
    status: 'active',
    current_period_end: '2026-03-02',
    cancel_at_period_end: false,
    analyses_used: 12,
    analyses_limit: 25,
  });
  const [isLoading, setIsLoading] = useState(false);

  const handleManageBilling = async () => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/subscription/portal', {
        method: 'POST',
      });
      if (res.ok) {
        const data = await res.json();
        window.location.href = data.url;
      }
    } catch (error) {
      console.error('Failed to open billing portal:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleUpgrade = async (priceId: string) => {
    setIsLoading(true);
    try {
      const res = await fetch('/api/v1/subscription/create-checkout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ price_id: priceId }),
      });
      if (res.ok) {
        const data = await res.json();
        window.location.href = data.url;
      }
    } catch (error) {
      console.error('Failed to create checkout session:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-900">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <Link href="/dashboard" className="text-blue-400 hover:underline text-sm mb-2 inline-block">
              ← Back to Dashboard
            </Link>
            <h1 className="text-3xl font-bold text-white">Subscription</h1>
            <p className="text-gray-400">Manage your LARUN subscription</p>
          </div>

          {/* Current Plan */}
          <Card className="p-6 mb-8">
            <h2 className="text-xl font-semibold text-white mb-4">Current Plan</h2>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-2xl font-bold text-white capitalize">{subscription.plan}</p>
                <p className="text-gray-400">
                  {subscription.status === 'active' ? (
                    <>Renews on {new Date(subscription.current_period_end).toLocaleDateString()}</>
                  ) : subscription.status === 'canceled' ? (
                    <>Expires on {new Date(subscription.current_period_end).toLocaleDateString()}</>
                  ) : (
                    <span className="text-yellow-400">Payment past due</span>
                  )}
                </p>
              </div>
              <Button
                variant="secondary"
                onClick={handleManageBilling}
                disabled={isLoading}
              >
                Manage Billing
              </Button>
            </div>

            {/* Usage */}
            <div className="mt-6 pt-6 border-t border-gray-700">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-gray-400">Analyses used this month</span>
                <span className="text-white">{subscription.analyses_used} / {subscription.analyses_limit}</span>
              </div>
              <div className="bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 rounded-full h-2"
                  style={{ width: `${(subscription.analyses_used / subscription.analyses_limit) * 100}%` }}
                />
              </div>
              <p className="text-gray-500 text-sm mt-2">
                {subscription.analyses_limit - subscription.analyses_used} analyses remaining
              </p>
            </div>
          </Card>

          {/* Upgrade Options */}
          {subscription.plan !== 'professional' && (
            <>
              <h2 className="text-xl font-semibold text-white mb-4">Upgrade Your Plan</h2>
              <div className="grid md:grid-cols-2 gap-6">
                {/* Annual Plan */}
                <Card className="p-6 border-blue-500/50">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">Annual Plan</h3>
                      <p className="text-gray-400 text-sm">Save $19/year</p>
                    </div>
                    <span className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs">BEST VALUE</span>
                  </div>
                  <p className="text-3xl font-bold text-white mb-1">
                    $89<span className="text-lg text-gray-400">/year</span>
                  </p>
                  <p className="text-gray-400 text-sm mb-4">~$7.42/month</p>
                  <ul className="space-y-2 mb-6">
                    {PLANS.hobbyist_annual.features.map((feature, i) => (
                      <li key={i} className="text-gray-300 text-sm flex items-center">
                        <span className="text-green-400 mr-2">✓</span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <Button
                    className="w-full"
                    onClick={() => handleUpgrade('price_hobbyist_annual')}
                    disabled={isLoading}
                  >
                    Switch to Annual
                  </Button>
                </Card>

                {/* Professional Plan */}
                <Card className="p-6 border-purple-500/50">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">Professional</h3>
                      <p className="text-gray-400 text-sm">For serious researchers</p>
                    </div>
                    <span className="bg-purple-500/20 text-purple-400 px-2 py-1 rounded text-xs">UNLIMITED</span>
                  </div>
                  <p className="text-3xl font-bold text-white mb-1">
                    $49<span className="text-lg text-gray-400">/month</span>
                  </p>
                  <p className="text-gray-400 text-sm mb-4">Unlimited analyses</p>
                  <ul className="space-y-2 mb-6">
                    {PLANS.professional.features.map((feature, i) => (
                      <li key={i} className="text-gray-300 text-sm flex items-center">
                        <span className="text-green-400 mr-2">✓</span>
                        {feature}
                      </li>
                    ))}
                  </ul>
                  <Button
                    className="w-full"
                    variant="secondary"
                    onClick={() => handleUpgrade('price_professional_monthly')}
                    disabled={isLoading}
                  >
                    Upgrade to Professional
                  </Button>
                </Card>
              </div>
            </>
          )}

          {/* FAQ */}
          <Card className="p-6 mt-8">
            <h2 className="text-xl font-semibold text-white mb-4">Frequently Asked Questions</h2>
            <div className="space-y-4">
              <div>
                <h3 className="text-white font-medium">Can I cancel anytime?</h3>
                <p className="text-gray-400 text-sm">
                  Yes, you can cancel your subscription at any time. You'll continue to have access
                  until the end of your billing period.
                </p>
              </div>
              <div>
                <h3 className="text-white font-medium">What happens to my analyses if I downgrade?</h3>
                <p className="text-gray-400 text-sm">
                  All your previous analyses and results remain accessible. You'll just be limited
                  to the lower plan's monthly analysis quota going forward.
                </p>
              </div>
              <div>
                <h3 className="text-white font-medium">Do unused analyses roll over?</h3>
                <p className="text-gray-400 text-sm">
                  No, unused analyses do not roll over to the next month. Each billing period
                  starts fresh with your plan's full allocation.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </main>

      <Footer />
    </div>
  );
}
