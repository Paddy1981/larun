import Stripe from 'stripe';

if (!process.env.STRIPE_SECRET_KEY) {
  throw new Error('STRIPE_SECRET_KEY is not set in environment variables');
}

export const stripe = new Stripe(process.env.STRIPE_SECRET_KEY, {
  apiVersion: '2024-12-18.acacia',
  typescript: true,
});

// Price IDs - these should be created in your Stripe dashboard
// and set as environment variables
export const PRICE_IDS = {
  monthly: process.env.STRIPE_PRICE_MONTHLY || 'price_monthly_placeholder',
  annual: process.env.STRIPE_PRICE_ANNUAL || 'price_annual_placeholder',
};

// Plan configurations
export const PLANS = {
  free: {
    name: 'Free',
    analyses: 3,
    features: ['3 analyses per month', 'Basic TinyML detection', 'CSV export'],
  },
  monthly: {
    name: 'Monthly',
    price: 9,
    analyses: 50,
    stripePriceId: PRICE_IDS.monthly,
    features: ['50 analyses per month', 'Advanced AI models', 'Priority processing', 'Email support'],
  },
  annual: {
    name: 'Annual',
    price: 89,
    analyses: -1, // unlimited
    stripePriceId: PRICE_IDS.annual,
    features: ['Unlimited analyses', 'All AI models + API', 'White-label reports', 'Priority support', '2 months free'],
  },
};

export type PlanType = keyof typeof PLANS;
