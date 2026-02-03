import { NextRequest, NextResponse } from 'next/server';
import crypto from 'crypto';

// Disable body parsing, we need the raw body for webhook verification
export const dynamic = 'force-dynamic';

// LemonSqueezy webhook event types
type LemonSqueezyEvent =
  | 'order_created'
  | 'order_refunded'
  | 'subscription_created'
  | 'subscription_updated'
  | 'subscription_cancelled'
  | 'subscription_resumed'
  | 'subscription_expired'
  | 'subscription_paused'
  | 'subscription_unpaused'
  | 'subscription_payment_success'
  | 'subscription_payment_failed'
  | 'subscription_payment_recovered';

interface WebhookPayload {
  meta: {
    event_name: LemonSqueezyEvent;
    custom_data?: {
      user_email?: string;
      plan?: string;
    };
  };
  data: {
    id: string;
    type: string;
    attributes: {
      status?: string;
      user_email?: string;
      customer_id?: number;
      variant_id?: number;
      product_id?: number;
      order_id?: number;
      renews_at?: string;
      ends_at?: string;
      [key: string]: unknown;
    };
  };
}

function verifySignature(payload: string, signature: string, secret: string): boolean {
  const hmac = crypto.createHmac('sha256', secret);
  const digest = hmac.update(payload).digest('hex');

  try {
    return crypto.timingSafeEqual(Buffer.from(signature), Buffer.from(digest));
  } catch {
    return false;
  }
}

export async function POST(request: NextRequest) {
  const body = await request.text();
  const signature = request.headers.get('x-signature');

  if (!signature) {
    return NextResponse.json(
      { error: 'Missing x-signature header' },
      { status: 400 }
    );
  }

  const webhookSecret = process.env.LEMONSQUEEZY_WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.error('LEMONSQUEEZY_WEBHOOK_SECRET is not set');
    return NextResponse.json(
      { error: 'Webhook secret not configured' },
      { status: 500 }
    );
  }

  // Verify signature
  if (!verifySignature(body, signature, webhookSecret)) {
    console.error('Webhook signature verification failed');
    return NextResponse.json(
      { error: 'Invalid signature' },
      { status: 400 }
    );
  }

  let payload: WebhookPayload;
  try {
    payload = JSON.parse(body);
  } catch {
    return NextResponse.json(
      { error: 'Invalid JSON payload' },
      { status: 400 }
    );
  }

  const eventName = payload.meta.event_name;
  const customData = payload.meta.custom_data;

  // Handle the event
  try {
    switch (eventName) {
      case 'order_created': {
        console.log('Order created:', {
          orderId: payload.data.id,
          userEmail: customData?.user_email || payload.data.attributes.user_email,
          plan: customData?.plan,
        });

        // TODO: Grant initial access to the user
        // await grantAccess(customData?.user_email, customData?.plan);
        break;
      }

      case 'subscription_created': {
        console.log('Subscription created:', {
          subscriptionId: payload.data.id,
          userEmail: customData?.user_email,
          plan: customData?.plan,
          status: payload.data.attributes.status,
        });

        // TODO: Update user's subscription in your database
        // await updateUserSubscription(customData?.user_email, {
        //   plan: customData?.plan,
        //   subscriptionId: payload.data.id,
        //   status: 'active',
        // });
        break;
      }

      case 'subscription_updated': {
        console.log('Subscription updated:', {
          subscriptionId: payload.data.id,
          status: payload.data.attributes.status,
          renewsAt: payload.data.attributes.renews_at,
        });

        // TODO: Update subscription status
        // await updateSubscriptionStatus(payload.data.id, payload.data.attributes.status);
        break;
      }

      case 'subscription_cancelled':
      case 'subscription_expired': {
        console.log('Subscription ended:', {
          subscriptionId: payload.data.id,
          event: eventName,
          endsAt: payload.data.attributes.ends_at,
        });

        // TODO: Handle subscription cancellation
        // await handleSubscriptionEnd(payload.data.id);
        break;
      }

      case 'subscription_payment_success': {
        console.log('Payment succeeded:', {
          subscriptionId: payload.data.id,
          customerId: payload.data.attributes.customer_id,
        });

        // TODO: Record successful payment, extend access
        // await recordPayment(payload.data.id, 'success');
        break;
      }

      case 'subscription_payment_failed': {
        console.log('Payment failed:', {
          subscriptionId: payload.data.id,
          customerId: payload.data.attributes.customer_id,
        });

        // TODO: Handle failed payment (send notification, update status)
        // await handleFailedPayment(payload.data.id);
        break;
      }

      case 'subscription_paused': {
        console.log('Subscription paused:', {
          subscriptionId: payload.data.id,
        });

        // TODO: Pause user access
        // await pauseSubscription(payload.data.id);
        break;
      }

      case 'subscription_resumed':
      case 'subscription_unpaused': {
        console.log('Subscription resumed:', {
          subscriptionId: payload.data.id,
        });

        // TODO: Resume user access
        // await resumeSubscription(payload.data.id);
        break;
      }

      default:
        console.log(`Unhandled event type: ${eventName}`);
    }

    return NextResponse.json({ received: true });
  } catch (error) {
    console.error('Error processing webhook:', error);
    return NextResponse.json(
      { error: 'Webhook handler failed' },
      { status: 500 }
    );
  }
}
