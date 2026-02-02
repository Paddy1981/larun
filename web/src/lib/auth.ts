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

export const authOptions: NextAuthOptions = {
  // No adapter - use JWT only for now (simpler, no database required)
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
  ],

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
