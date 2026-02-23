-- Add user_email column so NextAuth/Google OAuth users can save and query analyses
ALTER TABLE public.analyses
  ADD COLUMN IF NOT EXISTS user_email TEXT;

-- Make user_id nullable (NextAuth users don't have a Supabase UUID)
ALTER TABLE public.analyses
  ALTER COLUMN user_id DROP NOT NULL;

-- Drop the FK constraint (user_id may be a NextAuth sub, not a Supabase UUID)
ALTER TABLE public.analyses
  DROP CONSTRAINT IF EXISTS analyses_user_id_fkey;

-- Index for fast email-based queries
CREATE INDEX IF NOT EXISTS idx_analyses_user_email ON public.analyses(user_email);
