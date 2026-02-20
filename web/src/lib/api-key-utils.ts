/**
 * API Key utilities
 *
 * Keys look like:  lrn_live_<64 hex chars>
 * Only the SHA-256 hash is stored in the database.
 * The full key is shown to the user exactly once on creation.
 */
import crypto from 'crypto';

export const KEY_PREFIX = 'lrn_live_';

export function generateApiKey(): {
  /** Full key — return to user ONCE, never store */
  key: string;
  /** SHA-256 hash stored in DB */
  hash: string;
  /** First 16 chars shown in UI for identification */
  prefix: string;
} {
  const rand = crypto.randomBytes(32).toString('hex'); // 64 hex chars
  const key = `${KEY_PREFIX}${rand}`;
  const hash = crypto.createHash('sha256').update(key).digest('hex');
  const prefix = key.slice(0, KEY_PREFIX.length + 8); // e.g. "lrn_live_a1b2c3d4"
  return { key, hash, prefix };
}

export function hashApiKey(key: string): string {
  return crypto.createHash('sha256').update(key).digest('hex');
}

export function isValidKeyFormat(key: string): boolean {
  return typeof key === 'string' &&
    key.startsWith(KEY_PREFIX) &&
    key.length === KEY_PREFIX.length + 64;
}

/** Plan call limits (per month). -1 = unlimited. */
export const API_PLAN_LIMITS: Record<string, number> = {
  developer: 10_000,
  enterprise: -1,
  admin: -1,
};
