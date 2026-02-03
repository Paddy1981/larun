import { NextRequest, NextResponse } from 'next/server';
import { stripe } from '@/lib/stripe';
import Stripe from 'stripe';

// Disable body parsing, we need the raw body for webhook verification
export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  const body = await request.text();
  const signature = request.headers.get('stripe-signature');

  if (!signature) {
    return NextResponse.json(
      { error: 'Missing stripe-signature header' },
      { status: 400 }
    );
  }

  const webhookSecret = process.env.STRIPE_WEBHOOK_SECRET;
  if (!webhookSecret) {
    console.error('STRIPE_WEBHOOK_SECRET is not set');
    return NextResponse.json(
      { error: 'Webhook secret not configured' },
      { status: 500 }
    );
  }

  let event: Stripe.Event;

  try {
    event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
  } catch (err) {
    console.error('Webhook signature verification failed:', err);
    return NextResponse.json(
      { error: 'Webhook signature verification failed' },
      { status: 400 }
    );
  }

  // Handle the event
  try {
    switch (event.type) {
      case 'checkout.session.completed': {
        const session = event.data.object as Stripe.Checkout.Session;
        console.log('Checkout completed:', {
          customerId: session.customer,
          subscriptionId: session.subscription,
          userId: session.metadata?.userId,
          plan: session.metadata?.plan,
        });

        // TODO: Update user's subscription in your database
        // await updateUserSubscription(session.metadata?.userId, {
        //   plan: session.metadata?.plan,
        //   stripeCustomerId: session.customer,
        //   stripeSubscriptionId: session.subscription,
        //   status: 'active',
        // });
        break;
      }

      case 'customer.subscription.updated': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('Subscription updated:', {
          subscriptionId: subscription.id,
          status: subscription.status,
          customerId: subscription.customer,
          plan: subscription.metadata?.plan,
        });

        // TODO: Update subscription status in your database
        // await updateSubscriptionStatus(subscription.id, subscription.status);
        break;
      }

      case 'customer.subscription.deleted': {
        const subscription = event.data.object as Stripe.Subscription;
        console.log('Subscription canceled:', {
          subscriptionId: subscription.id,
          customerId: subscription.customer,
        });

        // TODO: Handle subscription cancellation
        // await cancelUserSubscription(subscription.metadata?.userId);
        break;
      }

      case 'invoice.payment_succeeded': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('Payment succeeded:', {
          invoiceId: invoice.id,
          customerId: invoice.customer,
          amountPaid: invoice.amount_paid,
        });

        // TODO: Record successful payment
        // await recordPayment(invoice);
        break;
      }

      case 'invoice.payment_failed': {
        const invoice = event.data.object as Stripe.Invoice;
        console.log('Payment failed:', {
          invoiceId: invoice.id,
          customerId: invoice.customer,
        });

        // TODO: Handle failed payment (send notification, update status)
        // await handleFailedPayment(invoice);
        break;
      }

      default:
        console.log(`Unhandled event type: ${event.type}`);
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
