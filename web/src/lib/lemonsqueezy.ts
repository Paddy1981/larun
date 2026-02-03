// LemonSqueezy API configuration
// Docs: https://docs.lemonsqueezy.com/api

const LEMONSQUEEZY_API_URL = 'https://api.lemonsqueezy.com/v1';

if (!process.env.LEMONSQUEEZY_API_KEY) {
  console.warn('LEMONSQUEEZY_API_KEY is not set in environment variables');
}

// Helper function for API requests
export async function lemonSqueezyApi(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const url = `${LEMONSQUEEZY_API_URL}${endpoint}`;

  return fetch(url, {
    ...options,
    headers: {
      'Accept': 'application/vnd.api+json',
      'Content-Type': 'application/vnd.api+json',
      'Authorization': `Bearer ${process.env.LEMONSQUEEZY_API_KEY}`,
      ...options.headers,
    },
  });
}

// Variant IDs - these are created in your LemonSqueezy dashboard
// A variant is a specific price/plan for a product
export const VARIANT_IDS = {
  monthly: process.env.LEMONSQUEEZY_VARIANT_MONTHLY || '',
  annual: process.env.LEMONSQUEEZY_VARIANT_ANNUAL || '',
};

// Store ID - your LemonSqueezy store
export const STORE_ID = process.env.LEMONSQUEEZY_STORE_ID || '';

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
    variantId: VARIANT_IDS.monthly,
    features: ['50 analyses per month', 'Advanced AI models', 'Priority processing', 'Email support'],
  },
  annual: {
    name: 'Annual',
    price: 89,
    analyses: -1, // unlimited
    variantId: VARIANT_IDS.annual,
    features: ['Unlimited analyses', 'All AI models + API', 'White-label reports', 'Priority support', '2 months free'],
  },
};

export type PlanType = keyof typeof PLANS;

// Create a checkout session
export async function createCheckout(params: {
  variantId: string;
  email: string;
  name?: string;
  customData?: Record<string, string>;
}): Promise<{ checkoutUrl: string } | { error: string }> {
  const { variantId, email, name, customData } = params;

  if (!variantId) {
    return { error: 'Variant ID not configured. Please set LEMONSQUEEZY_VARIANT_MONTHLY and LEMONSQUEEZY_VARIANT_ANNUAL.' };
  }

  if (!STORE_ID) {
    return { error: 'Store ID not configured. Please set LEMONSQUEEZY_STORE_ID.' };
  }

  try {
    const response = await lemonSqueezyApi('/checkouts', {
      method: 'POST',
      body: JSON.stringify({
        data: {
          type: 'checkouts',
          attributes: {
            checkout_data: {
              email,
              name: name || undefined,
              custom: customData || {},
            },
            checkout_options: {
              dark: false,
              embed: false,
              logo: true,
            },
            product_options: {
              redirect_url: `${process.env.NEXTAUTH_URL}/settings/subscription?success=true`,
              receipt_link_url: `${process.env.NEXTAUTH_URL}/settings/subscription`,
            },
          },
          relationships: {
            store: {
              data: {
                type: 'stores',
                id: STORE_ID,
              },
            },
            variant: {
              data: {
                type: 'variants',
                id: variantId,
              },
            },
          },
        },
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error('LemonSqueezy checkout error:', errorData);
      return { error: errorData.errors?.[0]?.detail || 'Failed to create checkout' };
    }

    const data = await response.json();
    return { checkoutUrl: data.data.attributes.url };
  } catch (error) {
    console.error('LemonSqueezy API error:', error);
    return { error: 'Failed to connect to payment service' };
  }
}

// Get customer portal URL
export async function getCustomerPortalUrl(customerId: string): Promise<{ portalUrl: string } | { error: string }> {
  try {
    const response = await lemonSqueezyApi(`/customers/${customerId}`);

    if (!response.ok) {
      const errorData = await response.json();
      return { error: errorData.errors?.[0]?.detail || 'Failed to get customer portal' };
    }

    const data = await response.json();
    const portalUrl = data.data.attributes.urls?.customer_portal;

    if (!portalUrl) {
      return { error: 'Customer portal URL not available' };
    }

    return { portalUrl };
  } catch (error) {
    console.error('LemonSqueezy API error:', error);
    return { error: 'Failed to connect to billing service' };
  }
}

// Verify webhook signature
export function verifyWebhookSignature(
  payload: string,
  signature: string,
  secret: string
): boolean {
  const crypto = require('crypto');
  const hmac = crypto.createHmac('sha256', secret);
  const digest = hmac.update(payload).digest('hex');
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(digest));
}
