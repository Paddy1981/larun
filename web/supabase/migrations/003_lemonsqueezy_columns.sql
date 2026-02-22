-- Migration 003: LemonSqueezy columns + email-keyed monthly quota
-- Apply via Supabase Dashboard SQL Editor

-- 1. Drop FK constraint on users.id → auth.users
--    (blocks NextAuth Google users who have no Supabase auth row)
ALTER TABLE public.users DROP CONSTRAINT IF EXISTS users_id_fkey;

-- 2. Add LemonSqueezy columns to users
ALTER TABLE public.users
  ADD COLUMN IF NOT EXISTS lemon_squeezy_customer_id TEXT,
  ADD COLUMN IF NOT EXISTS analyses_limit INTEGER NOT NULL DEFAULT 5;

-- 3. Add missing columns to subscriptions
ALTER TABLE public.subscriptions
  ADD COLUMN IF NOT EXISTS lemon_squeezy_subscription_id TEXT UNIQUE,
  ADD COLUMN IF NOT EXISTS lemon_squeezy_customer_id TEXT,
  ADD COLUMN IF NOT EXISTS variant_id TEXT,
  ADD COLUMN IF NOT EXISTS plan TEXT DEFAULT 'free',
  ADD COLUMN IF NOT EXISTS cancel_at_period_end BOOLEAN NOT NULL DEFAULT FALSE;

-- 4. Email-keyed monthly quota table (works for both Supabase Auth and NextAuth users)
CREATE TABLE IF NOT EXISTS public.monthly_quota (
  user_email TEXT NOT NULL,
  month      TEXT NOT NULL,  -- 'YYYY-MM'
  analyses_count INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (user_email, month)
);

-- 5. RLS on monthly_quota: deny all client access (service key bypasses RLS)
ALTER TABLE public.monthly_quota ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "deny_all_monthly_quota" ON public.monthly_quota;
CREATE POLICY "deny_all_monthly_quota" ON public.monthly_quota
  AS RESTRICTIVE
  FOR ALL
  USING (false);

-- 6. Atomic increment function (upserts count, safe for concurrent calls)
CREATE OR REPLACE FUNCTION public.increment_monthly_quota(p_email TEXT, p_month TEXT)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
  INSERT INTO public.monthly_quota (user_email, month, analyses_count)
  VALUES (p_email, p_month, 1)
  ON CONFLICT (user_email, month)
  DO UPDATE SET analyses_count = monthly_quota.analyses_count + 1;
END;
$$;
