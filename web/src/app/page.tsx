import Link from 'next/link';
import Header from '@/components/Header';
import PricingSection from '@/components/PricingSection';
import { HeroButtons, BottomCTA } from '@/components/HeroCTA';

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen bg-white text-[#202124]">
      {/* Header Navigation */}
      <Header />

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
            <strong>98% accuracy on real data.</strong>
          </p>
          <p className="text-lg text-[#1a73e8] font-medium mb-10">
            No PhD required.
          </p>
          <HeroButtons />
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
                Edge-optimized models achieving <strong className="text-[#202124]">98% accuracy</strong> on real Kepler/TESS data
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
      <PricingSection />

      {/* CTA Section */}
      <BottomCTA />

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
