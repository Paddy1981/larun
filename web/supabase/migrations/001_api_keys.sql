-- ============================================================
-- LARUN API Keys — run this in Supabase SQL Editor
-- Dashboard → SQL Editor → New query → paste → Run
-- ============================================================

-- Step 1: Add missing columns to existing table
-- (safe to run even if table was already partially created)
ALTER TABLE api_keys
  ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'developer',
  ADD COLUMN IF NOT EXISTS calls_this_month INTEGER NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS calls_limit INTEGER NOT NULL DEFAULT 10000,
  ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;

-- Step 2: Indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys (user_id);

-- Step 3: Enable RLS (idempotent)
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Step 4: Policies (drop and recreate to avoid conflicts)
DROP POLICY IF EXISTS "api_keys_owner_select" ON api_keys;
DROP POLICY IF EXISTS "api_keys_owner_insert" ON api_keys;
DROP POLICY IF EXISTS "api_keys_owner_update" ON api_keys;
DROP POLICY IF EXISTS "api_keys_owner_delete" ON api_keys;

CREATE POLICY "api_keys_owner_select" ON api_keys FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "api_keys_owner_insert" ON api_keys FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "api_keys_owner_update" ON api_keys FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "api_keys_owner_delete" ON api_keys FOR DELETE USING (auth.uid() = user_id);
