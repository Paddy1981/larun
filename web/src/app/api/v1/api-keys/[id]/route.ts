/**
 * DELETE /api/v1/api-keys/[id] — revoke an API key
 * PATCH  /api/v1/api-keys/[id] — rename a key
 */
import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@supabase/supabase-js';

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
const SUPABASE_ANON_KEY = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_KEY || SUPABASE_ANON_KEY;

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

interface RouteParams {
  params: Promise<{ id: string }>;
}

export async function DELETE(request: NextRequest, { params }: RouteParams) {
  const user = await getCallerUser(request);
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);

  const { error } = await service
    .from('api_keys')
    .update({ is_active: false })
    .eq('id', id)
    .eq('user_id', user.id); // ensure ownership

  if (error) {
    return NextResponse.json({ error: 'Failed to revoke key' }, { status: 500 });
  }

  return NextResponse.json({ success: true, message: 'API key revoked.' });
}

export async function PATCH(request: NextRequest, { params }: RouteParams) {
  const user = await getCallerUser(request);
  if (!user) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const { id } = await params;
  const body = await request.json().catch(() => ({}));
  const name = (body.name as string | undefined)?.slice(0, 80);
  if (!name) return NextResponse.json({ error: 'name is required' }, { status: 400 });

  const service = createClient(SUPABASE_URL, SUPABASE_SERVICE_KEY);
  const { data, error } = await service
    .from('api_keys')
    .update({ name })
    .eq('id', id)
    .eq('user_id', user.id)
    .select('id, name')
    .single();

  if (error) return NextResponse.json({ error: 'Failed to rename key' }, { status: 500 });
  return NextResponse.json(data);
}
