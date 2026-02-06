import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';
import { createServerSupabaseClient } from '@/lib/supabase';

// LemonSqueezy Variant IDs (maps variant ID to plan name)
const VARIANT_IDS = {
  '1278367': 'monthly',
  '1278370': 'annual',
} as const;

const PLAN_LIMITS = {
  free: 5,
  monthly: 50,
  annual: -1, // unlimited
};

// LemonSqueezy webhook event types
type WebhookEventName =
  | 'order_created'
  | 'order_refunded'
  | 'subscription_created'
  | 'subscription_updated'
  | 'subscription_cancelled'
  | 'subscription_resumed'
  | 'subscription_expired'
  | 'subscription_paused'
  | 'subscription_unpaused'
  | 'subscription_payment_failed'
  | 'subscription_payment_success'
  | 'subscription_payment_recovered'
  | 'subscription_payment_refunded'
  | 'subscription_plan_changed';

interface WebhookPayload {
  meta: {
    event_name: WebhookEventName;
    custom_data?: {
      user_id?: string;
    };
  };
  data: {
    id: string;
    type: string;
    attributes: {
      store_id: number;
      customer_id: number;
      order_id?: number;
      product_id?: number;
      variant_id?: number;
      status: string;
      status_formatted: string;
      user_email?: string;
      user_name?: string;
      created_at: string;
      updated_at: string;
      renews_at?: string;
      ends_at?: string;
      trial_ends_at?: string;
      [key: string]: unknown;
    };
  };
}

// Verify the webhook signature from LemonSqueezy
function verifySignature(payload: string, signature: string, secret: string): boolean {
  const hmac = crypto.createHmac('sha256', secret);
  const digest = hmac.update(payload).digest('hex');
  return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(digest));
}

export async function POST(request: NextRequest) {
  try {
    const webhookSecret = process.env.LEMONSQUEEZY_WEBHOOK_SECRET;

    if (!webhookSecret) {
      console.error('LEMONSQUEEZY_WEBHOOK_SECRET is not set');
      return NextResponse.json(
        { error: 'Webhook secret not configured' },
        { status: 500 }
      );
    }

    // Get the raw body and signature
    const rawBody = await request.text();
    const signature = request.headers.get('x-signature');

    if (!signature) {
      console.error('Missing X-Signature header');
      return NextResponse.json(
        { error: 'Missing signature' },
        { status: 401 }
      );
    }

    // Verify the signature
    if (!verifySignature(rawBody, signature, webhookSecret)) {
      console.error('Invalid webhook signature');
      return NextResponse.json(
        { error: 'Invalid signature' },
        { status: 401 }
      );
    }

    // Parse the payload
    const payload: WebhookPayload = JSON.parse(rawBody);
    const eventName = payload.meta.event_name;
    const customData = payload.meta.custom_data;
    const data = payload.data.attributes;

    console.log(`[LemonSqueezy Webhook] Event: ${eventName}`, {
      customerId: data.customer_id,
      status: data.status,
      userId: customData?.user_id,
    });

    // Initialize Supabase client
    const supabase = createServerSupabaseClient();

    // Handle different event types
    switch (eventName) {
      case 'subscription_created':
        await handleSubscriptionCreated(supabase, payload);
        break;

      case 'subscription_updated':
      case 'subscription_plan_changed':
        await handleSubscriptionUpdated(supabase, payload);
        break;

      case 'subscription_cancelled':
      case 'subscription_expired':
        await handleSubscriptionCancelled(supabase, payload);
        break;

      case 'subscription_resumed':
      case 'subscription_unpaused':
        await handleSubscriptionResumed(supabase, payload);
        break;

      case 'subscription_paused':
        await handleSubscriptionPaused(supabase, payload);
        break;

      case 'subscription_payment_failed':
        await handlePaymentFailed(supabase, payload);
        break;

      case 'subscription_payment_success':
      case 'subscription_payment_recovered':
        await handlePaymentSuccess(supabase, payload);
        break;

      default:
        console.log(`[LemonSqueezy Webhook] Unhandled event: ${eventName}`);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error('[LemonSqueezy Webhook] Error processing webhook:', error);
    return NextResponse.json(
      { error: 'Webhook processing failed' },
      { status: 500 }
    );
  }
}

// Helper to get plan from variant ID
function getPlanFromVariant(variantId: number | undefined): 'monthly' | 'annual' {
  const variantStr = String(variantId);
  return (VARIANT_IDS[variantStr as keyof typeof VARIANT_IDS] || 'monthly') as 'monthly' | 'annual';
}

// Event handlers

async function handleSubscriptionCreated(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;
  const plan = getPlanFromVariant(attrs.variant_id);

  console.log('[LemonSqueezy] Subscription created:', {
    subscriptionId: data.id,
    userId,
    email: attrs.user_email,
    plan,
  });

  if (!userId) {
    console.error('[LemonSqueezy] No user_id in custom_data');
    return;
  }

  // Create subscription record
  const { error: subError } = await supabase.from('subscriptions').insert({
    user_id: userId,
    lemon_squeezy_subscription_id: data.id,
    lemon_squeezy_customer_id: String(attrs.customer_id),
    variant_id: String(attrs.variant_id),
    plan,
    status: 'active',
    current_period_start: attrs.created_at,
    current_period_end: attrs.renews_at || attrs.created_at,
    cancel_at_period_end: false,
  });

  if (subError) {
    console.error('[LemonSqueezy] Error creating subscription:', subError);
  }

  // Update user's subscription tier
  const { error: userError } = await supabase.from('users').update({
    subscription_tier: plan,
    lemon_squeezy_customer_id: String(attrs.customer_id),
    analyses_limit: PLAN_LIMITS[plan],
  }).eq('id', userId);

  if (userError) {
    console.error('[LemonSqueezy] Error updating user:', userError);
  }
}

async function handleSubscriptionUpdated(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;
  const plan = getPlanFromVariant(attrs.variant_id);

  console.log('[LemonSqueezy] Subscription updated:', {
    subscriptionId: data.id,
    userId,
    status: attrs.status,
    plan,
  });

  // Update subscription record
  const { error: subError } = await supabase.from('subscriptions').update({
    variant_id: String(attrs.variant_id),
    plan,
    status: attrs.status === 'active' ? 'active' : attrs.status as string,
    current_period_end: attrs.renews_at,
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (subError) {
    console.error('[LemonSqueezy] Error updating subscription:', subError);
  }

  // Update user if we have user_id
  if (userId) {
    const { error: userError } = await supabase.from('users').update({
      subscription_tier: plan,
      analyses_limit: PLAN_LIMITS[plan],
    }).eq('id', userId);

    if (userError) {
      console.error('[LemonSqueezy] Error updating user:', userError);
    }
  }
}

async function handleSubscriptionCancelled(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Subscription cancelled:', {
    subscriptionId: data.id,
    userId,
    endsAt: attrs.ends_at,
  });

  // Update subscription status
  const { error: subError } = await supabase.from('subscriptions').update({
    status: 'cancelled',
    cancel_at_period_end: true,
    current_period_end: attrs.ends_at,
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (subError) {
    console.error('[LemonSqueezy] Error updating subscription:', subError);
  }

  // If subscription has expired, downgrade user to free
  if (payload.meta.event_name === 'subscription_expired' && userId) {
    const { error: userError } = await supabase.from('users').update({
      subscription_tier: 'free',
      analyses_limit: PLAN_LIMITS.free,
    }).eq('id', userId);

    if (userError) {
      console.error('[LemonSqueezy] Error downgrading user:', userError);
    }
  }
}

async function handleSubscriptionResumed(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;
  const plan = getPlanFromVariant(attrs.variant_id);

  console.log('[LemonSqueezy] Subscription resumed:', {
    subscriptionId: data.id,
    userId,
  });

  // Reactivate subscription
  const { error: subError } = await supabase.from('subscriptions').update({
    status: 'active',
    cancel_at_period_end: false,
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (subError) {
    console.error('[LemonSqueezy] Error updating subscription:', subError);
  }

  // Update user tier
  if (userId) {
    const { error: userError } = await supabase.from('users').update({
      subscription_tier: plan,
      analyses_limit: PLAN_LIMITS[plan],
    }).eq('id', userId);

    if (userError) {
      console.error('[LemonSqueezy] Error updating user:', userError);
    }
  }
}

async function handleSubscriptionPaused(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;

  console.log('[LemonSqueezy] Subscription paused:', {
    subscriptionId: data.id,
    userId,
  });

  // Mark subscription as paused
  const { error } = await supabase.from('subscriptions').update({
    status: 'paused',
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (error) {
    console.error('[LemonSqueezy] Error updating subscription:', error);
  }
}

async function handlePaymentFailed(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;

  console.log('[LemonSqueezy] Payment failed:', {
    subscriptionId: data.id,
    userId,
  });

  // Mark subscription as past_due
  const { error } = await supabase.from('subscriptions').update({
    status: 'past_due',
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (error) {
    console.error('[LemonSqueezy] Error updating subscription:', error);
  }
}

async function handlePaymentSuccess(supabase: ReturnType<typeof createServerSupabaseClient>, payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Payment successful:', {
    subscriptionId: data.id,
    userId,
    renewsAt: attrs.renews_at,
  });

  // Update subscription after successful payment
  const { error } = await supabase.from('subscriptions').update({
    status: 'active',
    current_period_end: attrs.renews_at,
  }).eq('lemon_squeezy_subscription_id', data.id);

  if (error) {
    console.error('[LemonSqueezy] Error updating subscription:', error);
  }
}
