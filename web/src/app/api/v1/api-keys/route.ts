/**
 * POST /api/v1/api-keys   — create a new API key
 * GET  /api/v1/api-keys   — list user's API keys
 *
 * Requires Supabase JWT in Authorization header:
 *   Authorization: Bearer <supabase_access_token>
 */
import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';
import { generateApiKey, API_PLAN_LIMITS } from '@/lib/api-key-utils';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || SUPABASE_ANON_KEY;

/** Validate the caller's Supabase JWT and return the authenticated user. */
async function getCallerUser(request: NextRequest) {
  const auth = request.headers.get('Authorization') || '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : '';
  if (!token) return null;

  const sb = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    global: { headers: { Authorization: `Bearer ${token}` } },
  });
  const { data: { user } } = await sb.auth.getUser();
  return user ?? null;
}

// ─── GET ─────────────────────────────────────────────────────────────────────

export async function GET(request: NextRequest) {
  const user = await getCallerUser(request);
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

  // Try full column set first; fall back to base schema if migration not applied yet
  let { data, error } = await service
    .from('api_keys')
    .select('id, key_prefix, name, plan, calls_this_month, calls_limit, last_used_at, created_at, is_active')
    .eq('user_id', user.id)
    .order('created_at', { ascending: false });

  if (error?.code === 'PGRST204') {
    // Migration not applied — query with base columns only
    const fallback = await service
      .from('api_keys')
      .select('id, key_prefix, name, last_used_at, created_at')
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });
    data = (fallback.data ?? []).map((k: Record<string, unknown>) => ({
      ...k, plan: 'developer', calls_this_month: 0, calls_limit: 10000, is_active: true,
    })) as typeof data;
    error = fallback.error;
  }

  if (error) {
    if (error.code === '42P01') {
      return NextResponse.json({
        keys: [],
        setup_required: true,
        message: 'Run supabase/migrations/001_api_keys.sql in the Supabase SQL Editor to enable API keys.',
      });
    }
    return NextResponse.json({ error: 'Failed to fetch API keys' }, { status: 500 });
  }

  return NextResponse.json({ keys: data ?? [] });
}

// ─── POST ────────────────────────────────────────────────────────────────────

export async function POST(request: NextRequest) {
  const user = await getCallerUser(request);
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const body = await request.json().catch(() => ({}));
  const name = (body.name as string | undefined)?.slice(0, 80) || 'Default Key';
  const plan = 'developer'; // all new keys start as developer

  // Check existing key count (max 5 per user)
  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  const { count } = await service
    .from('api_keys')
    .select('id', { count: 'exact', head: true })
    .eq('user_id', user.id)
    .eq('is_active', true);

  if ((count ?? 0) >= 5) {
    return NextResponse.json(
      { error: 'Maximum 5 active API keys allowed. Revoke an existing key first.' },
      { status: 429 }
    );
  }

  const { key, hash, prefix } = generateApiKey();
  const calls_limit = API_PLAN_LIMITS[plan] ?? 10_000;

  // Try full insert first (new schema); fall back to base schema if columns missing
  let insertData: Record<string, unknown> = {
    user_id: user.id,
    key_hash: hash,
    key_prefix: prefix,
    name,
    plan,
    calls_limit,
  };

  let result = await service
    .from('api_keys')
    .insert(insertData)
    .select('id, key_prefix, name, created_at')
    .single();

  if (result.error?.code === 'PGRST204') {
    // Columns not present yet — fall back to base schema
    insertData = { user_id: user.id, key_hash: hash, key_prefix: prefix, name };
    result = await service
      .from('api_keys')
      .insert(insertData)
      .select('id, key_prefix, name, created_at')
      .single();
  }

  if (result.error) {
    if (result.error.code === '42P01') {
      return NextResponse.json({
        error: 'API keys table not set up yet. Run supabase/migrations/001_api_keys.sql in the Supabase SQL Editor.',
        setup_required: true,
      }, { status: 503 });
    }
    console.error('Error creating API key:', result.error);
    return NextResponse.json({ error: 'Failed to create API key' }, { status: 500 });
  }

  // Return the full key ONCE — it is never retrievable again
  return NextResponse.json({
    ...(result.data ?? {}),
    plan: 'developer',
    calls_this_month: 0,
    calls_limit,
    key, // full secret — show to user now, never again
    message: 'Save this key — it will not be shown again.',
  }, { status: 201 });
}
