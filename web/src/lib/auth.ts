import { NextAuthOptions } from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';

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

// Get analyses limit based on subscription tier
function getAnalysesLimit(tier: string): number {
  switch (tier) {
    case 'professional':
      return -1; // Unlimited
    case 'hobbyist':
      return 25;
    default:
      return 3; // Free tier
  }
}

// Check if Google OAuth is configured
const googleClientId = process.env.GOOGLE_CLIENT_ID;
const googleClientSecret = process.env.GOOGLE_CLIENT_SECRET;
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

  secret: process.env.NEXTAUTH_SECRET,

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
        token.subscriptionTier = 'free';
        token.analysesThisMonth = 0;
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
