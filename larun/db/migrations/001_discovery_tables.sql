-- Migration 001: Discovery Engine Tables
-- Supports the Citizen Discovery Engine gamification system
-- Apply via: psql $DATABASE_URL < 001_discovery_tables.sql
--         or: Supabase Dashboard → SQL Editor

-- Enable UUID extension (already enabled in Supabase)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- discoveries: Records of submitted discovery candidates
-- ============================================================
CREATE TABLE IF NOT EXISTS discoveries (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    ra                  FLOAT NOT NULL,
    dec                 FLOAT NOT NULL,
    classification      VARCHAR(50),
    confidence          FLOAT,
    novelty_score       FLOAT,
    status              VARCHAR(20) DEFAULT 'candidate'
                            CHECK (status IN ('candidate', 'verified', 'rejected')),
    verification_count  INT DEFAULT 0,
    discovered_at       TIMESTAMPTZ DEFAULT NOW(),
    data_source         VARCHAR(20) CHECK (data_source IN ('tess', 'kepler', 'neowise', 'other')),
    models_used         JSONB DEFAULT '{}',
    priority            INT DEFAULT 0,
    -- Period info from PERIODOGRAM-001
    best_period         FLOAT,
    period_type         VARCHAR(30),
    -- Catalog match info
    catalog_known       BOOLEAN DEFAULT FALSE,
    catalog_matches     JSONB DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_discoveries_user_id ON discoveries(user_id);
CREATE INDEX IF NOT EXISTS idx_discoveries_status ON discoveries(status);
CREATE INDEX IF NOT EXISTS idx_discoveries_ra_dec ON discoveries(ra, dec);
-- Spatial index (PostGIS-style approximate using box)
CREATE INDEX IF NOT EXISTS idx_discoveries_coords ON discoveries(ra, dec)
    WHERE status = 'candidate';

-- ============================================================
-- verifications: Peer verification votes
-- ============================================================
CREATE TABLE IF NOT EXISTS verifications (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    discovery_id    UUID REFERENCES discoveries(id) ON DELETE CASCADE,
    verifier_id     UUID REFERENCES auth.users(id) ON DELETE SET NULL,
    verdict         VARCHAR(20) NOT NULL
                        CHECK (verdict IN ('confirm', 'reject', 'unsure')),
    notes           TEXT,
    verified_at     TIMESTAMPTZ DEFAULT NOW(),
    -- One vote per user per discovery
    UNIQUE (discovery_id, verifier_id)
);

CREATE INDEX IF NOT EXISTS idx_verifications_discovery ON verifications(discovery_id);

-- ============================================================
-- user_stats: Discovery achievements and gamification
-- ============================================================
CREATE TABLE IF NOT EXISTS user_stats (
    user_id                 UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    total_discoveries       INT DEFAULT 0,
    verified_discoveries    INT DEFAULT 0,
    total_analyses          INT DEFAULT 0,
    total_verifications     INT DEFAULT 0,
    rank                    VARCHAR(30) DEFAULT 'Stargazer',
    points                  INT DEFAULT 0,
    updated_at              TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Triggers: Auto-update verification_count on discoveries
-- ============================================================
CREATE OR REPLACE FUNCTION update_verification_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE discoveries
    SET verification_count = (
        SELECT COUNT(*) FROM verifications WHERE discovery_id = NEW.discovery_id
    )
    WHERE id = NEW.discovery_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_verification_count ON verifications;
CREATE TRIGGER trg_update_verification_count
    AFTER INSERT OR DELETE ON verifications
    FOR EACH ROW EXECUTE FUNCTION update_verification_count();

-- ============================================================
-- Row Level Security (Supabase)
-- ============================================================
ALTER TABLE discoveries ENABLE ROW LEVEL SECURITY;
ALTER TABLE verifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_stats ENABLE ROW LEVEL SECURITY;

-- Anyone can view verified discoveries and candidates
CREATE POLICY "Public read: verified discoveries"
    ON discoveries FOR SELECT
    USING (status IN ('verified', 'candidate'));

-- Authenticated users can submit discoveries
CREATE POLICY "Auth insert: discoveries"
    ON discoveries FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Authenticated users can verify candidates (but not their own)
CREATE POLICY "Auth insert: verifications"
    ON verifications FOR INSERT
    WITH CHECK (
        auth.uid() = verifier_id
        AND discovery_id IN (
            SELECT id FROM discoveries WHERE user_id != auth.uid()
        )
    );

-- Users can read their own stats
CREATE POLICY "Auth read: own stats"
    ON user_stats FOR SELECT
    USING (auth.uid() = user_id);
