/**
 * Pricing Page
 *
 * Displays subscription tiers with Stripe payment links
 */

import Link from 'next/link'
import { Check, Zap, Rocket, Shield } from 'lucide-react'

export default function PricingPage() {
  const tiers = [
    {
      name: 'Free',
      price: '$0',
      period: 'forever',
      description: 'For getting started',
      icon: <Zap className="w-6 h-6" />,
      features: [
        '5 analyses per month',
        'Basic TinyML detection',
        'CSV export',
        'Community support',
        'FITS file upload',
      ],
      cta: 'Get Started Free',
      ctaLink: '/cloud/auth/signup',
      highlighted: false,
      isExternal: false,
    },
    {
      name: 'Monthly',
      price: '$9',
      period: 'per month',
      description: 'For active users',
      icon: <Rocket className="w-6 h-6" />,
      features: [
        '50 analyses per month',
        'Advanced AI models',
        'Priority processing',
        'JSON & CSV export',
        'Email support',
        'Cloud storage',
      ],
      cta: 'Subscribe Now',
      ctaLink: 'https://larunspace.lemonsqueezy.com/checkout/buy/f35b9320-79ed-462d-bdf7-1ec4841eadbb',
      highlighted: true,
      badge: 'Popular',
      isExternal: true,
    },
    {
      name: 'Annual',
      price: '$89',
      period: 'per year',
      description: 'Best value',
      icon: <Shield className="w-6 h-6" />,
      features: [
        'Unlimited analyses',
        'All AI models + API',
        'White-label reports',
        'Priority support',
        'Advanced features',
        'Bulk processing',
      ],
      cta: 'Subscribe Now',
      ctaLink: 'https://larunspace.lemonsqueezy.com/checkout/buy/ff35095c-0eac-427c-8309-8d55448979a2',
      highlighted: false,
      badge: 'Best value',
      isExternal: true,
    },
  ]

  return (
    <div className="pt-24 pb-16 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-5xl mb-6">Pricing</h1>
          <p className="text-xl text-larun-medium-gray max-w-2xl mx-auto">
            Choose the plan that fits your research needs. All plans include access to our 8 TinyML models.
          </p>
        </div>

        {/* Pricing Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {tiers.map((tier) => (
            <div
              key={tier.name}
              className={`rounded-lg border-2 p-8 relative ${
                tier.highlighted
                  ? 'border-larun-black shadow-xl scale-105'
                  : 'border-larun-light-gray'
              }`}
            >
              {/* Badge */}
              {tier.badge && (
                <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-larun-black text-larun-white px-4 py-1 rounded-full text-xs font-medium">
                  {tier.badge}
                </div>
              )}

              {/* Header */}
              <div className="mb-6">
                <div className="w-12 h-12 rounded-lg bg-larun-black text-larun-white flex items-center justify-center mb-4">
                  {tier.icon}
                </div>
                <h3 className="text-2xl font-medium mb-2">{tier.name}</h3>
                <p className="text-sm text-larun-medium-gray mb-4">
                  {tier.description}
                </p>
                <div className="flex items-baseline gap-2">
                  <span className="text-5xl font-medium">{tier.price}</span>
                  <span className="text-larun-medium-gray">{tier.period}</span>
                </div>
              </div>

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-3">
                    <Check className="w-5 h-5 text-larun-black flex-shrink-0 mt-0.5" />
                    <span className="text-sm text-larun-dark-gray">{feature}</span>
                  </li>
                ))}
              </ul>

              {/* CTA */}
              {tier.isExternal ? (
                <a
                  href={tier.ctaLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className={`btn w-full justify-center ${
                    tier.highlighted ? 'btn-primary' : 'btn-outline'
                  }`}
                >
                  {tier.cta}
                </a>
              ) : (
                <Link
                  href={tier.ctaLink}
                  className={`btn w-full justify-center ${
                    tier.highlighted ? 'btn-primary' : 'btn-outline'
                  }`}
                >
                  {tier.cta}
                </Link>
              )}
            </div>
          ))}
        </div>

        {/* FAQ */}
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl text-center mb-12">Frequently Asked Questions</h2>

          <div className="space-y-6">
            <div>
              <h4 className="text-lg font-medium mb-2">What counts as an analysis?</h4>
              <p className="text-larun-medium-gray">
                Each FITS file you upload and analyze with a TinyML model counts as one analysis.
                Running the same file through multiple models counts as multiple analyses.
              </p>
            </div>

            <div>
              <h4 className="text-lg font-medium mb-2">Can I upgrade or downgrade anytime?</h4>
              <p className="text-larun-medium-gray">
                Yes! You can upgrade or downgrade your plan at any time. Changes take effect immediately,
                and we'll prorate any charges.
              </p>
            </div>

            <div>
              <h4 className="text-lg font-medium mb-2">What payment methods do you accept?</h4>
              <p className="text-larun-medium-gray">
                We accept all major credit cards (Visa, Mastercard, American Express, Discover) via Stripe.
                Enterprise customers can pay via invoice.
              </p>
            </div>

            <div>
              <h4 className="text-lg font-medium mb-2">Is there a free trial?</h4>
              <p className="text-larun-medium-gray">
                Our Free tier is always available with 100 analyses per month. No credit card required.
                You can upgrade to Pro or Enterprise anytime.
              </p>
            </div>

            <div>
              <h4 className="text-lg font-medium mb-2">What happens if I exceed my quota?</h4>
              <p className="text-larun-medium-gray">
                If you reach your monthly analysis limit, you'll be prompted to upgrade. Your existing
                analyses and data remain accessible. Quotas reset on the 1st of each month.
              </p>
            </div>

            <div>
              <h4 className="text-lg font-medium mb-2">Do you offer academic discounts?</h4>
              <p className="text-larun-medium-gray">
                Yes! We offer 50% discounts for students, researchers, and educational institutions.
                Contact us at <a href="mailto:sales@larun.space" className="text-larun-black underline">sales@larun.space</a> with
                your academic email.
              </p>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-16 text-center">
          <p className="text-larun-medium-gray mb-4">
            Need help choosing a plan? We're here to help.
          </p>
          <Link href="mailto:sales@larun.space" className="btn btn-secondary">
            Contact Sales
          </Link>
        </div>
      </div>
    </div>
  )
}
