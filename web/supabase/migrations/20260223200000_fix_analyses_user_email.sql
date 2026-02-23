-- Re-apply user_email column addition (previous migration was recorded but did not execute)
ALTER TABLE public.analyses ADD COLUMN IF NOT EXISTS user_email TEXT;
CREATE INDEX IF NOT EXISTS idx_analyses_user_email ON public.analyses(user_email);
