'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';

export default function Header() {
  const { data: session, status } = useSession();
  const [showUserMenu, setShowUserMenu] = useState(false);

  const getUserInitial = () => {
    if (session?.user?.name) {
      return session.user.name.charAt(0).toUpperCase();
    }
    if (session?.user?.email) {
      return session.user.email.charAt(0).toUpperCase();
    }
    return '?';
  };

  return (
    <header className="fixed top-0 left-0 right-0 bg-white border-b border-[#dadce0] z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3">
            <span className="text-2xl font-bold text-[#202124]">Larun<span className="text-[#1a73e8]">.</span></span>
            <span className="text-sm font-medium text-[#5f6368]">AstroTinyML</span>
          </Link>

          {/* Center Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            <Link href="#features" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
              Features
            </Link>
            <Link href="/models" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
              Models
            </Link>
            <Link href="/cloud" className="text-[#1a73e8] hover:text-[#1557b0] text-sm font-semibold transition-colors">
              Cloud ☁️
            </Link>
            <Link href="/guide" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
              Guide
            </Link>
            <Link href="#pricing" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
              Pricing
            </Link>
            <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
              GitHub
            </a>
          </nav>

          {/* Auth Section */}
          <div className="flex items-center gap-3">
            {status === 'loading' ? (
              <div className="w-8 h-8 rounded-full bg-[#f1f3f4] animate-pulse"></div>
            ) : session ? (
              <>
                <Link
                  href="/dashboard"
                  className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors"
                >
                  Dashboard
                </Link>
                {/* User Menu */}
                <div className="relative">
                  <button
                    onClick={() => setShowUserMenu(!showUserMenu)}
                    className="w-8 h-8 bg-[#1a73e8] rounded-full flex items-center justify-center text-white text-sm font-medium cursor-pointer hover:bg-[#1557b0] transition-colors"
                  >
                    {session.user?.image ? (
                      <img src={session.user.image} alt="" className="w-8 h-8 rounded-full" />
                    ) : (
                      getUserInitial()
                    )}
                  </button>
                  {showUserMenu && (
                    <>
                      <div
                        className="fixed inset-0 z-40"
                        onClick={() => setShowUserMenu(false)}
                      ></div>
                      <div className="absolute right-0 mt-2 w-72 bg-white rounded-lg shadow-lg border border-[#dadce0] py-2 z-50">
                        <div className="px-4 py-3 border-b border-[#dadce0]">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-[#1a73e8] rounded-full flex items-center justify-center text-white font-medium">
                              {session.user?.image ? (
                                <img src={session.user.image} alt="" className="w-10 h-10 rounded-full" />
                              ) : (
                                getUserInitial()
                              )}
                            </div>
                            <div>
                              <p className="text-sm font-medium text-[#202124]">{session.user?.name || 'User'}</p>
                              <p className="text-xs text-[#5f6368]">{session.user?.email}</p>
                            </div>
                          </div>
                        </div>
                        <div className="py-1">
                          <Link
                            href="/dashboard"
                            className="block px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]"
                            onClick={() => setShowUserMenu(false)}
                          >
                            Dashboard
                          </Link>
                          <Link
                            href="/settings/subscription"
                            className="block px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]"
                            onClick={() => setShowUserMenu(false)}
                          >
                            Usage & Billing
                          </Link>
                          <button
                            onClick={() => signOut({ callbackUrl: '/' })}
                            className="w-full text-left px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]"
                          >
                            Sign out
                          </button>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </>
            ) : (
              <>
                <Link
                  href="/auth/login"
                  className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors"
                >
                  Sign In
                </Link>
                <Link
                  href="/dashboard"
                  className="bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium px-5 py-2.5 rounded-lg transition-colors"
                >
                  Try Demo
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
