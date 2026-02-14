'use client'

/**
 * Simple Navigation Component (no external icons)
 */

import Link from 'next/link'
import { usePathname } from 'next/navigation'

export function Navigation() {
  const pathname = usePathname()

  const isActive = (path: string) => pathname === path

  return (
    <nav className="fixed top-0 left-0 right-0 bg-white border-b border-larun-light-gray z-50">
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <Link href="/" className="text-xl font-medium hover:opacity-80 transition-opacity">
            <span className="text-larun-black">LARUN</span>
            <span className="text-larun-medium-gray">.</span>
            <span className="text-larun-medium-gray">SPACE</span>
          </Link>

          {/* Navigation Links */}
          <div className="flex items-center gap-8">
            <Link
              href="/"
              className={`text-sm font-medium transition-colors ${
                isActive('/')
                  ? 'text-larun-black'
                  : 'text-larun-medium-gray hover:text-larun-black'
              }`}
            >
              Home
            </Link>

            <Link
              href="/pricing"
              className={`text-sm font-medium transition-colors ${
                isActive('/pricing')
                  ? 'text-larun-black'
                  : 'text-larun-medium-gray hover:text-larun-black'
              }`}
            >
              Pricing
            </Link>

            <Link
              href="/analyze"
              className={`text-sm font-medium transition-colors ${
                isActive('/analyze')
                  ? 'text-larun-black'
                  : 'text-larun-medium-gray hover:text-larun-black'
              }`}
            >
              Analyze
            </Link>

            <Link
              href="/dashboard"
              className={`text-sm font-medium transition-colors ${
                isActive('/dashboard')
                  ? 'text-larun-black'
                  : 'text-larun-medium-gray hover:text-larun-black'
              }`}
            >
              Dashboard
            </Link>

            {/* Auth Buttons */}
            <div className="flex items-center gap-3 ml-4 pl-4 border-l border-larun-light-gray">
              <Link
                href="/auth/login"
                className="text-sm font-medium text-larun-medium-gray hover:text-larun-black transition-colors"
              >
                Sign In
              </Link>

              <Link
                href="/auth/signup"
                className="bg-larun-black text-white px-4 py-2 rounded text-sm font-medium hover:bg-larun-dark-gray transition-colors"
              >
                Get Started
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
