-- Migration: 002_admin_subscription_gate
-- Description: Activate admin subscription for padhu.veera@gmail.com
--              Uses Lemon Squeezy column schema.

DO $$
DECLARE
  v_user_id UUID;
BEGIN
  SELECT id INTO v_user_id
  FROM public.users
  WHERE email = 'padhu.veera@gmail.com'
  LIMIT 1;

  IF v_user_id IS NULL THEN
    RAISE NOTICE 'Admin user padhu.veera@gmail.com not found — sign up first, then re-run.';
    RETURN;
  END IF;

  IF EXISTS (
    SELECT 1 FROM public.subscriptions
    WHERE user_id = v_user_id AND status = 'active'
  ) THEN
    RAISE NOTICE 'Admin already has an active subscription.';
    RETURN;
  END IF;

  INSERT INTO public.subscriptions (
    user_id,
    lemon_squeezy_subscription_id,
    lemon_squeezy_customer_id,
    variant_id,
    plan,
    tier,
    status,
    current_period_start,
    current_period_end,
    cancel_at_period_end
  ) VALUES (
    v_user_id,
    'manual_admin_' || v_user_id::TEXT,
    'admin',
    '0',
    'annual',
    'annual',
    'active',
    NOW(),
    NOW() + INTERVAL '100 years',
    FALSE
  );

  UPDATE public.users
  SET subscription_tier = 'annual',
      analyses_limit = -1
  WHERE id = v_user_id;

  RAISE NOTICE 'Admin subscription activated for padhu.veera@gmail.com';
END;
$$;
