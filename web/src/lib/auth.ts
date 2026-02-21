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

// Fetch subscription tier from Supabase using service key (server-side only)
async function fetchSubscriptionTier(email: string): Promise<string> {
  const url = process.env.NEXT_PUBLIC_SUPABASE_URL || process.env.SUPABASE_URL || '';
  const key = process.env.SUPABASE_SERVICE_KEY || process.env.SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || '';
  if (!url || !key) return 'free';
  try {
    const sb = createClient(url, key);
    const { data } = await sb
      .from('users')
      .select('subscription_tier, analyses_limit')
      .eq('email', email)
      .single();
    if (data?.subscription_tier) return data.subscription_tier;
  } catch {
    // ignore
  }
  return 'free';
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
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
        token.analysesThisMonth = 0;
        // Look up subscription tier from Supabase on first sign-in
        if (user.email) {
          const dbTier = await fetchSubscriptionTier(user.email);
          token.subscriptionTier = toNextAuthTier(dbTier);
        } else {
          token.subscriptionTier = 'free';
        }
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
