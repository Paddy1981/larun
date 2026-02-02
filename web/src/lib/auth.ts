import { NextAuthOptions } from 'next-auth';
import { SupabaseAdapter } from '@auth/supabase-adapter';
import GoogleProvider from 'next-auth/providers/google';
import GitHubProvider from 'next-auth/providers/github';
import EmailProvider from 'next-auth/providers/email';
import CredentialsProvider from 'next-auth/providers/credentials';
import { createServerSupabaseClient } from './supabase';
import jwt from 'jsonwebtoken';

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
  adapter: SupabaseAdapter({
    url: process.env.NEXT_PUBLIC_SUPABASE_URL!,
    secret: process.env.SUPABASE_SERVICE_ROLE_KEY!,
  }),

  providers: [
    // Google OAuth
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
      allowDangerousEmailAccountLinking: true,
    }),

    // GitHub OAuth
    GitHubProvider({
      clientId: process.env.GITHUB_CLIENT_ID!,
      clientSecret: process.env.GITHUB_CLIENT_SECRET!,
      allowDangerousEmailAccountLinking: true,
    }),

    // Magic link email authentication
    EmailProvider({
      server: {
        host: process.env.SMTP_HOST,
        port: Number(process.env.SMTP_PORT) || 587,
        auth: {
          user: process.env.SMTP_USER,
          pass: process.env.SMTP_PASS,
        },
      },
      from: process.env.EMAIL_FROM || 'LARUN <noreply@larun.space>',
    }),

    // Email/Password credentials (optional fallback)
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        const supabase = createServerSupabaseClient();

        const { data, error } = await supabase.auth.signInWithPassword({
          email: credentials.email,
          password: credentials.password,
        });

        if (error || !data.user) {
          return null;
        }

        // Get user profile from our users table
        const { data: profile } = await supabase
          .from('users')
          .select('*')
          .eq('id', data.user.id)
          .single();

        return {
          id: data.user.id,
          email: data.user.email!,
          name: profile?.name || data.user.user_metadata?.name,
          image: profile?.image || data.user.user_metadata?.avatar_url,
          subscriptionTier: profile?.subscription_tier || 'free',
          analysesThisMonth: profile?.analyses_this_month || 0,
        };
      },
    }),
  ],

  session: {
    strategy: 'jwt',
    maxAge: 30 * 24 * 60 * 60, // 30 days
  },

  pages: {
    signIn: '/auth/login',
    signUp: '/auth/register',
    error: '/auth/error',
    verifyRequest: '/auth/verify',
  },

  callbacks: {
    async signIn({ user, account }) {
      // Create or update user in our users table on OAuth sign in
      if (account?.provider !== 'credentials' && user.email) {
        const supabase = createServerSupabaseClient();

        const { data: existingUser } = await supabase
          .from('users')
          .select('id')
          .eq('email', user.email)
          .single();

        if (!existingUser) {
          // Create new user record
          await supabase.from('users').insert({
            id: user.id,
            email: user.email,
            name: user.name,
            image: user.image,
            subscription_tier: 'free',
            analyses_this_month: 0,
          });
        }
      }
      return true;
    },

    async jwt({ token, user, account }) {
      // Initial sign in
      if (user) {
        token.subscriptionTier = user.subscriptionTier || 'free';
        token.analysesThisMonth = user.analysesThisMonth || 0;
      }

      // Generate Supabase access token for API calls
      if (account?.provider && token.sub) {
        const payload = {
          aud: 'authenticated',
          exp: Math.floor(Date.now() / 1000) + 60 * 60, // 1 hour
          sub: token.sub,
          email: token.email,
          role: 'authenticated',
        };
        token.supabaseAccessToken = jwt.sign(
          payload,
          process.env.SUPABASE_JWT_SECRET!
        );
      }

      return token;
    },

    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.sub!;
        session.user.subscriptionTier = token.subscriptionTier || 'free';
        session.user.analysesThisMonth = token.analysesThisMonth || 0;
        session.user.analysesLimit = getAnalysesLimit(token.subscriptionTier || 'free');
        session.supabaseAccessToken = token.supabaseAccessToken;
      }
      return session;
    },
  },

  events: {
    async signIn({ user }) {
      console.log(`User signed in: ${user.email}`);
    },
    async signOut({ token }) {
      console.log(`User signed out: ${token?.email}`);
    },
  },

  debug: process.env.NODE_ENV === 'development',
};
