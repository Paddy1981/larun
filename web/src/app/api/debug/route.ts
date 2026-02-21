import { NextResponse } from 'next/server';

/**
 * GET /api/debug
 *
 * Debug endpoint to check auth configuration.
 * Only shows whether env vars are set, not their values.
 */
export async function GET() {
  const sk = process.env.SUPABASE_SERVICE_KEY || '';
  const config = {
    GOOGLE_CLIENT_ID: !!process.env.GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET: !!process.env.GOOGLE_CLIENT_SECRET,
    NEXTAUTH_SECRET: !!process.env.NEXTAUTH_SECRET,
    NEXTAUTH_URL: process.env.NEXTAUTH_URL || 'not set',
    NODE_ENV: process.env.NODE_ENV,
    SUPABASE_URL: process.env.NEXT_PUBLIC_SUPABASE_URL?.slice(0, 40) || 'not set',
    SUPABASE_SERVICE_KEY_SET: !!sk,
    SUPABASE_SERVICE_KEY_LEN: sk.length,
    SUPABASE_SERVICE_KEY_FIRST20: sk.slice(0, 20),
    SUPABASE_SERVICE_KEY_LAST10: sk.slice(-10),
    configured: !!(
      process.env.GOOGLE_CLIENT_ID &&
      process.env.GOOGLE_CLIENT_SECRET &&
      process.env.NEXTAUTH_SECRET
    ),
  };

  return NextResponse.json(config);
}
