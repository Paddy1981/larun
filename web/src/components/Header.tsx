'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useSession, signOut } from 'next-auth/react';

export default function Header() {
  const { data: session, status } = useSession();
  const [showUserMenu, setShowUserMenu]       = useState(false);
  const [showResourcesMenu, setShowResourcesMenu] = useState(false);
  const [showMobile, setShowMobile]           = useState(false);

  const getUserInitial = () => {
    if (session?.user?.name)  return session.user.name.charAt(0).toUpperCase();
    if (session?.user?.email) return session.user.email.charAt(0).toUpperCase();
    return '?';
  };

  return (
    <>
      <header className="fixed top-0 left-0 right-0 bg-white border-b border-[#dadce0] z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">

            {/* Logo */}
            <Link href="/" className="flex items-center gap-2 shrink-0">
              <span className="text-xl font-bold text-[#202124]">
                Larun<span className="text-[#1a73e8]">.</span><span className="text-[#1a73e8]">Space</span>
              </span>
            </Link>

            {/* Desktop nav */}
            <nav className="hidden md:flex items-center gap-1">

              <Link href="/models"
                className="px-3 py-2 text-sm text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4] rounded-lg font-medium transition-colors">
                Models
              </Link>

              <Link href="/discover"
                className="px-3 py-2 text-sm text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4] rounded-lg font-medium transition-colors">
                Discover
              </Link>

              <Link href="/leaderboard"
                className="px-3 py-2 text-sm text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4] rounded-lg font-medium transition-colors">
                Leaderboard
              </Link>

              <a href="https://sattrack.larun.space" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-1.5 px-3 py-2 text-sm text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4] rounded-lg font-medium transition-colors">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <circle cx="12" cy="12" r="10" />
                  <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" strokeLinecap="round" />
                </svg>
                SatTrack
              </a>

              <Link href="/cloud/pricing"
                className="px-3 py-2 text-sm text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4] rounded-lg font-medium transition-colors">
                Pricing
              </Link>

              {/* Resources dropdown */}
              <div className="relative"
                onMouseEnter={() => setShowResourcesMenu(true)}
                onMouseLeave={() => setShowResourcesMenu(false)}>
                <button className={`flex items-center gap-1 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${showResourcesMenu ? 'text-[#202124] bg-[#f1f3f4]' : 'text-[#5f6368] hover:text-[#202124] hover:bg-[#f1f3f4]'}`}>
                  Resources
                  <svg className={`w-3.5 h-3.5 transition-transform ${showResourcesMenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {showResourcesMenu && (
                  <div className="absolute left-0 top-full pt-1 z-50 w-52">
                    <div className="bg-white border border-[#dadce0] rounded-xl shadow-lg py-1.5 overflow-hidden">

                      <p className="text-[10px] font-semibold text-[#9aa0a6] uppercase tracking-wider px-4 pt-2 pb-1">Documentation</p>
                      <Link href="/guide" onClick={() => setShowResourcesMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4] transition-colors">
                        <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z" />
                        </svg>
                        User Guide
                      </Link>
                      <Link href="/faq" onClick={() => setShowResourcesMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4] transition-colors">
                        <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z" />
                        </svg>
                        FAQ
                      </Link>

                      <div className="h-px bg-[#f1f3f4] my-1.5" />

                      <p className="text-[10px] font-semibold text-[#9aa0a6] uppercase tracking-wider px-4 pt-1 pb-1">Standards</p>
                      <Link href="/ecss" onClick={() => setShowResourcesMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4] transition-colors">
                        <span className="w-5 h-5 bg-[#1a73e8] rounded text-white text-[10px] flex items-center justify-center font-bold shrink-0">E</span>
                        ECSS Standards
                      </Link>
                      <Link href="/nasa" onClick={() => setShowResourcesMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4] transition-colors">
                        <span className="w-5 h-5 bg-[#0b3d91] rounded text-white text-[10px] flex items-center justify-center font-bold shrink-0">N</span>
                        NASA Standards
                      </Link>

                      <div className="h-px bg-[#f1f3f4] my-1.5" />

                      <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer"
                        onClick={() => setShowResourcesMenu(false)}
                        className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4] transition-colors">
                        <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
                        </svg>
                        GitHub
                      </a>
                    </div>
                  </div>
                )}
              </div>
            </nav>

            {/* Right side: Cloud CTA + auth */}
            <div className="flex items-center gap-2">

              {/* Cloud CTA — always visible */}
              <Link href="/cloud/analyze"
                className="hidden sm:flex items-center gap-1.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path d="M3 15a4 4 0 004 4h9a5 5 0 10-.1-9.999 5.002 5.002 0 10-9.78 2.096A4.001 4.001 0 003 15z" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Cloud
              </Link>

              {/* Auth */}
              {status === 'loading' ? (
                <div className="w-8 h-8 rounded-full bg-[#f1f3f4] animate-pulse" />
              ) : session ? (
                <div className="relative">
                  <button onClick={() => setShowUserMenu(v => !v)}
                    className="w-9 h-9 bg-[#1a73e8] rounded-full flex items-center justify-center text-white text-sm font-medium overflow-hidden hover:ring-2 hover:ring-[#1a73e8] hover:ring-offset-1 transition-all">
                    {session.user?.image
                      ? <img src={session.user.image} alt="" className="w-9 h-9 rounded-full" />
                      : getUserInitial()}
                  </button>
                  {showUserMenu && (
                    <>
                      <div className="fixed inset-0 z-40" onClick={() => setShowUserMenu(false)} />
                      <div className="absolute right-0 mt-2 w-64 bg-white rounded-xl shadow-lg border border-[#dadce0] py-2 z-50">
                        <div className="px-4 py-3 border-b border-[#f1f3f4]">
                          <p className="text-sm font-medium text-[#202124]">{session.user?.name || 'User'}</p>
                          <p className="text-xs text-[#5f6368]">{session.user?.email}</p>
                        </div>
                        <Link href="/dashboard" onClick={() => setShowUserMenu(false)}
                          className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]">
                          <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z" />
                          </svg>
                          Dashboard
                        </Link>
                        <Link href="/cloud/analyze" onClick={() => setShowUserMenu(false)}
                          className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]">
                          <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z" />
                          </svg>
                          Run Analysis
                        </Link>
                        <Link href="/cloud/billing" onClick={() => setShowUserMenu(false)}
                          className="flex items-center gap-2.5 px-4 py-2 text-sm text-[#3c4043] hover:bg-[#f1f3f4]">
                          <svg className="w-4 h-4 text-[#5f6368]" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M20 4H4c-1.11 0-2 .89-2 2v12c0 1.11.89 2 2 2h16c1.11 0 2-.89 2-2V6c0-1.11-.89-2-2-2zm0 14H4v-6h16v6zm0-10H4V6h16v2z" />
                          </svg>
                          Billing
                        </Link>
                        <div className="h-px bg-[#f1f3f4] my-1" />
                        <button onClick={() => signOut({ callbackUrl: '/' })}
                          className="w-full flex items-center gap-2.5 px-4 py-2 text-sm text-[#c5221f] hover:bg-[#fce8e6]">
                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M17 7l-1.41 1.41L18.17 11H8v2h10.17l-2.58 2.58L17 17l5-5zM4 5h8V3H4c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h8v-2H4V5z" />
                          </svg>
                          Sign out
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <Link href="/auth/login"
                  className="text-sm text-[#5f6368] hover:text-[#202124] font-medium px-3 py-2 rounded-lg hover:bg-[#f1f3f4] transition-colors">
                  Sign in
                </Link>
              )}

              {/* Mobile hamburger */}
              <button
                onClick={() => setShowMobile(v => !v)}
                className="md:hidden w-10 h-10 flex items-center justify-center rounded-lg hover:bg-[#f1f3f4] transition-colors"
              >
                <svg className="w-5 h-5 text-[#5f6368]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  {showMobile
                    ? <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                    : <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />}
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        {showMobile && (
          <div className="md:hidden border-t border-[#dadce0] bg-white px-4 py-3 space-y-1">
            <MobileLink href="/cloud/analyze" onClick={() => setShowMobile(false)} primary>Cloud Analyze</MobileLink>
            <MobileLink href="/discover"      onClick={() => setShowMobile(false)}>Discover</MobileLink>
            <MobileLink href="/leaderboard"   onClick={() => setShowMobile(false)}>Leaderboard</MobileLink>
            <MobileLink href="/models"        onClick={() => setShowMobile(false)}>Models</MobileLink>
            <a href="https://sattrack.larun.space" target="_blank" rel="noopener noreferrer"
              onClick={() => setShowMobile(false)}
              className="block px-3 py-2.5 text-sm text-[#3c4043] hover:bg-[#f1f3f4] rounded-lg font-medium">
              SatTrack ↗
            </a>
            <MobileLink href="/cloud/pricing" onClick={() => setShowMobile(false)}>Pricing</MobileLink>
            <div className="h-px bg-[#f1f3f4] my-1" />
            <MobileLink href="/guide"          onClick={() => setShowMobile(false)}>User Guide</MobileLink>
            <MobileLink href="/faq"            onClick={() => setShowMobile(false)}>FAQ</MobileLink>
            <MobileLink href="/ecss"           onClick={() => setShowMobile(false)}>ECSS Standards</MobileLink>
            <MobileLink href="/nasa"           onClick={() => setShowMobile(false)}>NASA Standards</MobileLink>
            <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer"
              onClick={() => setShowMobile(false)}
              className="block px-3 py-2.5 text-sm text-[#3c4043] hover:bg-[#f1f3f4] rounded-lg font-medium">
              GitHub ↗
            </a>
            {session && (
              <>
                <div className="h-px bg-[#f1f3f4] my-1" />
                <MobileLink href="/dashboard" onClick={() => setShowMobile(false)}>Dashboard</MobileLink>
                <button onClick={() => { signOut({ callbackUrl: '/' }); setShowMobile(false); }}
                  className="w-full text-left block px-3 py-2.5 text-sm text-[#c5221f] hover:bg-[#fce8e6] rounded-lg font-medium">
                  Sign out
                </button>
              </>
            )}
          </div>
        )}
      </header>
    </>
  );
}

function MobileLink({ href, children, onClick, primary }: { href: string; children: React.ReactNode; onClick: () => void; primary?: boolean }) {
  return (
    <Link href={href} onClick={onClick}
      className={`block px-3 py-2.5 text-sm rounded-lg font-medium transition-colors ${
        primary
          ? 'bg-[#1a73e8] text-white hover:bg-[#1557b0]'
          : 'text-[#3c4043] hover:bg-[#f1f3f4]'
      }`}>
      {children}
    </Link>
  );
}
