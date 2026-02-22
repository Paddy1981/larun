import { NextAuthOptions } from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';
import { createClient } from '@supabase/supabase-js';

// Extend the session type to include custom fields
declare module 'next-auth' {
  interface Session {
    user: {
      id: string;
      email: string;
      name?: string | null;
      image?: string | null;
      subscriptionTier: 'free' | 'hobbyist' | 'professional';
      analysesThisMonth: number;
      analysesLimit: number;
    };
    supabaseAccessToken?: string;
  }

  interface User {
    subscriptionTier?: 'free' | 'hobbyist' | 'professional';
    analysesThisMonth?: number;
  }
}

declare module 'next-auth/jwt' {
  interface JWT {
    subscriptionTier?: 'free' | 'hobbyist' | 'professional';
    analysesThisMonth?: number;
    supabaseAccessToken?: string;
  }
}

// Map Supabase subscription_tier → NextAuth tier label
function toNextAuthTier(dbTier: string): 'free' | 'hobbyist' | 'professional' {
  if (dbTier === 'annual') return 'professional';
  if (dbTier === 'monthly') return 'hobbyist';
  if (dbTier === 'professional') return 'professional';
  if (dbTier === 'hobbyist') return 'hobbyist';
  return 'free';
}

// Get analyses limit based on subscription tier
function getAnalysesLimit(tier: string): number {
  switch (tier) {
    case 'professional':
    case 'annual':
      return -1; // Unlimited
    case 'hobbyist':
    case 'monthly':
      return 50;
    default:
      return 5; // Free tier
  }
}

// Fetch live subscription tier + current month usage from DB
async function fetchUserData(email: string): Promise<{ tier: string; analysesThisMonth: number }> {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!supabaseUrl || !serviceKey) {
    return { tier: 'free', analysesThisMonth: 0 };
  }
  const sb = createClient(supabaseUrl, serviceKey);
  const currentMonth = new Date().toISOString().slice(0, 7);
  const [userResult, quotaResult] = await Promise.all([
    sb.from('users').select('subscription_tier').eq('email', email).maybeSingle(),
    sb.from('monthly_quota').select('analyses_count').eq('user_email', email).eq('month', currentMonth).maybeSingle(),
  ]);
  return {
    tier: userResult.data?.subscription_tier || 'free',
    analysesThisMonth: quotaResult.data?.analyses_count ?? 0,
  };
}

// Check if Google OAuth is configured
// .trim() guards against trailing newlines injected by Vercel's env var storage
const googleClientId = process.env.GOOGLE_CLIENT_ID?.trim();
const googleClientSecret = process.env.GOOGLE_CLIENT_SECRET?.trim();
const isGoogleConfigured = !!(googleClientId && googleClientSecret);

// Log configuration status (helpful for debugging)
if (typeof window === 'undefined') {
  console.log('[NextAuth] Google OAuth configured:', isGoogleConfigured);
  console.log('[NextAuth] GOOGLE_CLIENT_ID set:', !!googleClientId);
  console.log('[NextAuth] GOOGLE_CLIENT_SECRET set:', !!googleClientSecret);
  console.log('[NextAuth] NEXTAUTH_SECRET set:', !!process.env.NEXTAUTH_SECRET);
  console.log('[NextAuth] NEXTAUTH_URL:', process.env.NEXTAUTH_URL || 'not set');
}

export const authOptions: NextAuthOptions = {
  // No adapter - use JWT only for now (simpler, no database required)
  providers: isGoogleConfigured
    ? [
        GoogleProvider({
          clientId: googleClientId!,
          clientSecret: googleClientSecret!,
        }),
      ]
    : [],

  secret: process.env.NEXTAUTH_SECRET?.trim(),

  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },

  pages: {
    signIn: '/auth/login',
    error: '/auth/error',
  },

  callbacks: {
    async jwt({ token, user, trigger }) {
      // Refresh on first sign-in OR on explicit update trigger
      const email = (user?.email || token.email) as string | undefined;
      if (email && (user || trigger === 'update')) {
        const { tier, analysesThisMonth } = await fetchUserData(email);
        token.subscriptionTier = toNextAuthTier(tier);
        token.analysesThisMonth = analysesThisMonth;
        if (user) token.id = user.id;
      }
      return token;
    },

    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.id as string || token.sub!;
        session.user.subscriptionTier = (token.subscriptionTier as 'free' | 'hobbyist' | 'professional') || 'free';
        session.user.analysesThisMonth = (token.analysesThisMonth as number) || 0;
        session.user.analysesLimit = getAnalysesLimit(token.subscriptionTier as string || 'free');
      }
      return session;
    },
  },

  debug: process.env.NODE_ENV === 'development',
};
