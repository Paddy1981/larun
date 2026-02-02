import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen bg-white text-[#202124]">
      {/* Header Navigation */}
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
              <Link href="#how-it-works" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
                How It Works
              </Link>
              <Link href="#pricing" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
                Pricing
              </Link>
              <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] hover:text-[#202124] text-sm font-medium transition-colors">
                GitHub
              </a>
            </nav>

            {/* Auth Buttons */}
            <div className="flex items-center gap-3">
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
            </div>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 bg-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 bg-[#f1f3f4] text-[#5f6368] text-sm font-medium px-4 py-2 rounded-full mb-8">
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/>
            </svg>
            Federation of TinyML for Space Science
          </div>

          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-[#202124] mb-6 leading-tight">
            Discover exoplanets<br />with AI
          </h1>
          <p className="text-xl text-[#5f6368] mb-3 max-w-2xl mx-auto">
            Analyze NASA TESS and Kepler data using TinyML-powered transit detection.
          </p>
          <p className="text-xl text-[#202124] mb-3">
            <strong>81.8% accuracy.</strong>
          </p>
          <p className="text-lg text-[#1a73e8] font-medium mb-10">
            No PhD required.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/auth/register"
              className="bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium px-8 py-3.5 rounded-lg transition-colors text-base"
            >
              Start Exploring - Free
            </Link>
            <Link
              href="/dashboard"
              className="bg-white hover:bg-[#f1f3f4] text-[#202124] font-medium px-8 py-3.5 rounded-lg border border-[#dadce0] transition-colors text-base"
            >
              View Demo
            </Link>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 bg-[#f1f3f4]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[#202124] text-center mb-4">
            How It Works
          </h2>
          <p className="text-[#5f6368] text-center mb-12 max-w-2xl mx-auto">
            From natural language to exoplanet discovery in four simple steps
          </p>

          <div className="grid md:grid-cols-4 gap-6">
            {/* Step 1 */}
            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-10 h-10 bg-[#202124] text-white rounded-full flex items-center justify-center font-bold text-lg mb-4">
                1
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Chat with AI</h3>
              <p className="text-[#5f6368] text-sm">
                Use natural language to describe what you want to analyze. No coding required.
              </p>
            </div>

            {/* Step 2 */}
            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-10 h-10 bg-[#202124] text-white rounded-full flex items-center justify-center font-bold text-lg mb-4">
                2
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Fetch NASA Data</h3>
              <p className="text-[#5f6368] text-sm">
                Automatically retrieve light curve data from NASA TESS and Kepler missions.
              </p>
            </div>

            {/* Step 3 */}
            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-10 h-10 bg-[#202124] text-white rounded-full flex items-center justify-center font-bold text-lg mb-4">
                3
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Detect Transits</h3>
              <p className="text-[#5f6368] text-sm">
                TinyML algorithms analyze the data to identify planetary transit signals.
              </p>
            </div>

            {/* Step 4 */}
            <div className="bg-white rounded-xl p-6 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-10 h-10 bg-[#202124] text-white rounded-full flex items-center justify-center font-bold text-lg mb-4">
                4
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Generate Reports</h3>
              <p className="text-[#5f6368] text-sm">
                Get publication-ready reports with visualizations and orbital parameters.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[#202124] text-center mb-4">
            Features
          </h2>
          <p className="text-[#5f6368] text-center mb-12 max-w-2xl mx-auto">
            Professional-grade tools for exoplanet detection
          </p>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 - TinyML Detection */}
            <div className="text-center p-6">
              <div className="w-14 h-14 bg-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">TinyML Detection</h3>
              <p className="text-[#5f6368] text-sm">
                Edge-optimized models achieving <strong className="text-[#202124]">81.8% accuracy</strong> in transit detection
              </p>
            </div>

            {/* Feature 2 - NASA Data */}
            <div className="text-center p-6">
              <div className="w-14 h-14 border-2 border-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-[#202124]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">NASA TESS/Kepler Data</h3>
              <p className="text-[#5f6368] text-sm">
                Direct integration with NASA mission archives for seamless data access
              </p>
            </div>

            {/* Feature 3 - BLS Periodogram */}
            <div className="text-center p-6">
              <div className="w-14 h-14 bg-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">BLS Periodogram</h3>
              <p className="text-[#5f6368] text-sm">
                Box Least Squares analysis for precise period determination
              </p>
            </div>

            {/* Feature 4 - Automated Reports */}
            <div className="text-center p-6">
              <div className="w-14 h-14 border-2 border-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-[#202124]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Automated Reports</h3>
              <p className="text-[#5f6368] text-sm">
                Generate publication-ready reports with visualizations
              </p>
            </div>

            {/* Feature 5 - Habitable Zone */}
            <div className="text-center p-6">
              <div className="w-14 h-14 bg-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Habitable Zone</h3>
              <p className="text-[#5f6368] text-sm">
                Automatic evaluation of planetary habitability potential
              </p>
            </div>

            {/* Feature 6 - AI Chat */}
            <div className="text-center p-6">
              <div className="w-14 h-14 border-2 border-[#202124] rounded-xl flex items-center justify-center mx-auto mb-4">
                <svg className="w-7 h-7 text-[#202124]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-[#202124] mb-2">AI Chat Interface</h3>
              <p className="text-[#5f6368] text-sm">
                Conversational interface for intuitive data exploration
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Target Audience Section */}
      <section className="py-16 bg-[#f1f3f4]">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-2xl font-bold text-[#202124] mb-8">
            Built For
          </h2>
          <div className="flex flex-wrap justify-center gap-3">
            <span className="bg-white text-[#202124] px-5 py-2.5 rounded-full text-sm font-medium shadow-sm border border-[#dadce0]">
              Researchers
            </span>
            <span className="bg-white text-[#202124] px-5 py-2.5 rounded-full text-sm font-medium shadow-sm border border-[#dadce0]">
              Students
            </span>
            <span className="bg-white text-[#202124] px-5 py-2.5 rounded-full text-sm font-medium shadow-sm border border-[#dadce0]">
              Amateur Astronomers
            </span>
            <span className="bg-white text-[#202124] px-5 py-2.5 rounded-full text-sm font-medium shadow-sm border border-[#dadce0]">
              Educators
            </span>
            <span className="bg-white text-[#202124] px-5 py-2.5 rounded-full text-sm font-medium shadow-sm border border-[#dadce0]">
              Space Enthusiasts
            </span>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 bg-white">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[#202124] text-center mb-4">
            Simple Pricing
          </h2>
          <p className="text-[#5f6368] text-center mb-12">
            Start free, upgrade when you need more
          </p>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Free */}
            <div className="bg-white border border-[#dadce0] rounded-xl p-6">
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Free</h3>
              <p className="text-[#5f6368] text-sm mb-4">For getting started</p>
              <div className="text-3xl font-bold text-[#202124] mb-6">$0</div>
              <ul className="space-y-3 mb-6 text-sm text-[#5f6368]">
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  3 analyses per month
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Basic TinyML detection
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  CSV export
                </li>
              </ul>
              <Link href="/auth/register" className="block text-center bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium py-2.5 rounded-lg transition-colors">
                Get Started
              </Link>
            </div>

            {/* Monthly */}
            <div className="bg-[#1a73e8] text-white rounded-xl p-6 relative">
              <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-[#202124] text-white text-xs font-medium px-3 py-1 rounded-full">
                Popular
              </div>
              <h3 className="text-lg font-semibold mb-2">Monthly</h3>
              <p className="text-blue-100 text-sm mb-4">For active users</p>
              <div className="text-3xl font-bold mb-6">$9<span className="text-lg font-normal">/mo</span></div>
              <ul className="space-y-3 mb-6 text-sm">
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  50 analyses per month
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Advanced AI models
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Priority processing
                </li>
              </ul>
              <Link href="/auth/register" className="block text-center bg-white hover:bg-[#f1f3f4] text-[#1a73e8] font-medium py-2.5 rounded-lg transition-colors">
                Subscribe
              </Link>
            </div>

            {/* Annual */}
            <div className="bg-white border border-[#dadce0] rounded-xl p-6">
              <h3 className="text-lg font-semibold text-[#202124] mb-2">Annual</h3>
              <p className="text-[#5f6368] text-sm mb-4">Best value</p>
              <div className="text-3xl font-bold text-[#202124] mb-6">$89<span className="text-lg font-normal text-[#5f6368]">/yr</span></div>
              <ul className="space-y-3 mb-6 text-sm text-[#5f6368]">
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  Unlimited analyses
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  All AI models + API
                </li>
                <li className="flex items-center gap-2">
                  <svg className="w-4 h-4 text-[#1a73e8]" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                  White-label reports
                </li>
              </ul>
              <Link href="/auth/register" className="block text-center bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium py-2.5 rounded-lg transition-colors">
                Subscribe
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-[#202124] text-white">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Start Exploring the Cosmos
          </h2>
          <p className="text-gray-400 mb-8 max-w-2xl mx-auto">
            Join researchers and astronomy enthusiasts discovering new worlds with AI-powered analysis. No credit card required.
          </p>
          <Link
            href="/auth/register"
            className="inline-block bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium px-8 py-3.5 rounded-lg transition-colors"
          >
            Create Free Account
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold text-[#202124]">Larun<span className="text-[#1a73e8]">.</span></span>
              <span className="text-sm text-[#5f6368]">AstroTinyML</span>
            </Link>

            {/* Links */}
            <div className="flex items-center gap-6 text-sm text-[#5f6368]">
              <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                Larun Engineering
              </a>
              <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                GitHub
              </a>
              <Link href="#" className="hover:text-[#202124] transition-colors">
                Privacy
              </Link>
              <Link href="#" className="hover:text-[#202124] transition-colors">
                Terms
              </Link>
            </div>

            {/* Copyright */}
            <p className="text-sm text-[#5f6368]">
              &copy; {new Date().getFullYear()} Larun Engineering. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
