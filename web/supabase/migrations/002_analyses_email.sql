-- 002_analyses_email.sql
-- Allow email-based user identification so NextAuth Google OAuth users
-- (whose sub IDs are not Supabase UUIDs) can still save analyses.

-- Add user_email column for email-based lookups
ALTER TABLE public.analyses
  ADD COLUMN IF NOT EXISTS user_email TEXT;

-- Make user_id nullable so analyses can be saved without a Supabase UUID
ALTER TABLE public.analyses
  ALTER COLUMN user_id DROP NOT NULL;

-- Drop the FK constraint (user_id may be a NextAuth/Google sub, not Supabase UUID)
ALTER TABLE public.analyses
  DROP CONSTRAINT IF EXISTS analyses_user_id_fkey;

-- Index for fast email-based queries
CREATE INDEX IF NOT EXISTS idx_analyses_user_email ON public.analyses(user_email);

-- RLS: users can read/write their own rows by email
DROP POLICY IF EXISTS "analyses_email_select" ON public.analyses;
CREATE POLICY "analyses_email_select" ON public.analyses
  FOR SELECT USING (
    user_email = current_setting('request.jwt.claims', true)::json->>'email'
    OR user_id::text = (current_setting('request.jwt.claims', true)::json->>'sub')
  );
