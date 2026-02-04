import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';

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

    // Handle different event types
    switch (eventName) {
      case 'order_created':
        await handleOrderCreated(payload);
        break;

      case 'subscription_created':
        await handleSubscriptionCreated(payload);
        break;

      case 'subscription_updated':
      case 'subscription_plan_changed':
        await handleSubscriptionUpdated(payload);
        break;

      case 'subscription_cancelled':
      case 'subscription_expired':
        await handleSubscriptionCancelled(payload);
        break;

      case 'subscription_resumed':
      case 'subscription_unpaused':
        await handleSubscriptionResumed(payload);
        break;

      case 'subscription_paused':
        await handleSubscriptionPaused(payload);
        break;

      case 'subscription_payment_failed':
        await handlePaymentFailed(payload);
        break;

      case 'subscription_payment_success':
      case 'subscription_payment_recovered':
        await handlePaymentSuccess(payload);
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

// Event handlers - integrate with your database here

async function handleOrderCreated(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;

  console.log('[LemonSqueezy] Order created:', {
    orderId: data.id,
    userId,
    email: data.attributes.user_email,
  });

  // TODO: Update user's subscription in database
  // Example with Supabase:
  // await supabase.from('users').update({
  //   subscription_status: 'active',
  //   lemon_squeezy_customer_id: data.attributes.customer_id,
  // }).eq('id', userId);
}

async function handleSubscriptionCreated(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Subscription created:', {
    subscriptionId: data.id,
    userId,
    status: attrs.status,
    renewsAt: attrs.renews_at,
  });

  // TODO: Store subscription details in database
  // await supabase.from('subscriptions').insert({
  //   user_id: userId,
  //   lemon_squeezy_subscription_id: data.id,
  //   status: attrs.status,
  //   current_period_end: attrs.renews_at,
  //   variant_id: attrs.variant_id,
  // });
}

async function handleSubscriptionUpdated(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Subscription updated:', {
    subscriptionId: data.id,
    userId,
    status: attrs.status,
    renewsAt: attrs.renews_at,
  });

  // TODO: Update subscription in database
  // await supabase.from('subscriptions').update({
  //   status: attrs.status,
  //   current_period_end: attrs.renews_at,
  //   variant_id: attrs.variant_id,
  // }).eq('lemon_squeezy_subscription_id', data.id);
}

async function handleSubscriptionCancelled(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Subscription cancelled:', {
    subscriptionId: data.id,
    userId,
    endsAt: attrs.ends_at,
  });

  // TODO: Mark subscription as cancelled
  // await supabase.from('subscriptions').update({
  //   status: 'cancelled',
  //   ends_at: attrs.ends_at,
  // }).eq('lemon_squeezy_subscription_id', data.id);
}

async function handleSubscriptionResumed(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Subscription resumed:', {
    subscriptionId: data.id,
    userId,
    status: attrs.status,
  });

  // TODO: Reactivate subscription
  // await supabase.from('subscriptions').update({
  //   status: 'active',
  //   ends_at: null,
  // }).eq('lemon_squeezy_subscription_id', data.id);
}

async function handleSubscriptionPaused(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;

  console.log('[LemonSqueezy] Subscription paused:', {
    subscriptionId: data.id,
    userId,
  });

  // TODO: Mark subscription as paused
  // await supabase.from('subscriptions').update({
  //   status: 'paused',
  // }).eq('lemon_squeezy_subscription_id', data.id);
}

async function handlePaymentFailed(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;

  console.log('[LemonSqueezy] Payment failed:', {
    subscriptionId: data.id,
    userId,
  });

  // TODO: Handle failed payment (send email, update status)
  // await supabase.from('subscriptions').update({
  //   status: 'past_due',
  // }).eq('lemon_squeezy_subscription_id', data.id);
}

async function handlePaymentSuccess(payload: WebhookPayload) {
  const { data, meta } = payload;
  const userId = meta.custom_data?.user_id;
  const attrs = data.attributes;

  console.log('[LemonSqueezy] Payment successful:', {
    subscriptionId: data.id,
    userId,
    renewsAt: attrs.renews_at,
  });

  // TODO: Update subscription after successful payment
  // await supabase.from('subscriptions').update({
  //   status: 'active',
  //   current_period_end: attrs.renews_at,
  // }).eq('lemon_squeezy_subscription_id', data.id);
}
