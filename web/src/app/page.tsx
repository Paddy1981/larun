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

      {/* Cloud Platform Highlight */}
      <section className="py-16 bg-gradient-to-r from-[#1a73e8] to-[#174ea6] text-white">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center gap-8">
            <div className="flex-1">
              <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm px-4 py-2 rounded-full text-sm font-medium mb-4">
                <span>✨</span>
                <span>NEW: Cloud Platform</span>
              </div>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                TinyML Inference in the Cloud
              </h2>
              <p className="text-white/90 text-lg mb-6">
                Upload FITS files and run real-time inference with our 8 specialized TinyML models.
                Get instant classifications with 98% accuracy—no setup required.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center gap-3">
                  <svg className="w-5 h-5 text-white flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  <span>5 free analyses per month</span>
                </li>
                <li className="flex items-center gap-3">
                  <svg className="w-5 h-5 text-white flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  <span>8 specialized astronomy models (exoplanets, variable stars, flares)</span>
                </li>
                <li className="flex items-center gap-3">
                  <svg className="w-5 h-5 text-white flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  <span>Inference in &lt;100ms with TensorFlow Lite</span>
                </li>
              </ul>
              <div className="flex flex-col sm:flex-row gap-4">
                <Link
                  href="/cloud"
                  className="inline-block bg-white text-[#1a73e8] hover:bg-gray-50 font-medium px-6 py-3 rounded-lg transition-colors text-center"
                >
                  Try Cloud Platform →
                </Link>
                <Link
                  href="/cloud/pricing"
                  className="inline-block bg-white/10 hover:bg-white/20 backdrop-blur-sm text-white font-medium px-6 py-3 rounded-lg transition-colors text-center border border-white/30"
                >
                  View Pricing
                </Link>
              </div>
            </div>
            <div className="flex-1 hidden md:block">
              <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-8 border border-white/20">
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"/>
                      </svg>
                    </div>
                    <div>
                      <p className="font-semibold">Upload FITS File</p>
                      <p className="text-sm text-white/70">Drag & drop your light curve</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                      </svg>
                    </div>
                    <div>
                      <p className="font-semibold">Select Model</p>
                      <p className="text-sm text-white/70">Choose from 8 TinyML models</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 bg-white/20 rounded-lg flex items-center justify-center">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                      </svg>
                    </div>
                    <div>
                      <p className="font-semibold">Get Results</p>
                      <p className="text-sm text-white/70">Instant classification & confidence</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SatTrack Section */}
      <section className="py-16 bg-white border-t border-[#dadce0]">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center gap-10">
            {/* Visual side */}
            <div className="flex-1 hidden md:flex items-center justify-center">
              <div className="relative w-64 h-64">
                {/* Globe ring */}
                <div className="absolute inset-0 rounded-full border-2 border-[#e8f0fe] flex items-center justify-center">
                  <div className="w-48 h-48 rounded-full border border-[#c5d5f8] flex items-center justify-center">
                    <div className="w-32 h-32 rounded-full bg-[#e8f0fe] flex items-center justify-center">
                      <svg className="w-16 h-16 text-[#1a73e8]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
                  </div>
                </div>
                {/* Orbit dots */}
                <div className="absolute top-4 left-1/2 -translate-x-1/2 w-3 h-3 bg-[#1a73e8] rounded-full shadow-lg"></div>
                <div className="absolute bottom-8 right-6 w-2 h-2 bg-[#fbbc04] rounded-full shadow-lg"></div>
                <div className="absolute top-1/2 left-2 w-2 h-2 bg-[#34a853] rounded-full shadow-lg"></div>
              </div>
            </div>

            {/* Content side */}
            <div className="flex-1">
              <div className="inline-flex items-center gap-2 bg-[#e8f0fe] text-[#1a73e8] text-sm font-medium px-4 py-2 rounded-full mb-4">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 3l14 9-14 9V3z" />
                </svg>
                Live Now
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-[#202124] mb-4">
                Track Satellites in Real Time
              </h2>
              <p className="text-[#5f6368] text-lg mb-6">
                SatTrack is a real-time 3D satellite tracker. Monitor 2,000+ satellites, predict ISS and Starlink passes over your location, and view live space weather—all in your browser.
              </p>
              <ul className="space-y-3 mb-8">
                <li className="flex items-center gap-3 text-[#3c4043]">
                  <svg className="w-5 h-5 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  2,000+ satellites tracked live with TLE data
                </li>
                <li className="flex items-center gap-3 text-[#3c4043]">
                  <svg className="w-5 h-5 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  Pass predictions for ISS, Starlink & more
                </li>
                <li className="flex items-center gap-3 text-[#3c4043]">
                  <svg className="w-5 h-5 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  Interactive 3D globe view in your browser
                </li>
                <li className="flex items-center gap-3 text-[#3c4043]">
                  <svg className="w-5 h-5 text-[#1a73e8] flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                  </svg>
                  Live space weather & solar activity data
                </li>
              </ul>
              <a
                href="https://sattrack.larun.space"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium px-6 py-3 rounded-lg transition-colors"
              >
                Open SatTrack
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                </svg>
              </a>
            </div>
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
            Professional-grade tools for astronomical analysis powered by 8 specialized TinyML models
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

      {/* Knowledge Hubs Section */}
      <section className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[#202124] text-center mb-4">
            Standards Knowledge Hubs
          </h2>
          <p className="text-[#5f6368] text-center mb-12 max-w-2xl mx-auto">
            Comprehensive reference libraries for space engineering standards — instantly searchable, always accessible.
          </p>
          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {/* ECSS Hub */}
            <Link href="/ecss" className="group block bg-white border border-[#dadce0] rounded-2xl p-8 hover:shadow-lg hover:border-[#1a73e8] transition-all duration-200">
              <div className="flex items-center gap-4 mb-5">
                <div className="w-14 h-14 bg-[#1a73e8] rounded-xl flex items-center justify-center text-white font-bold text-xl flex-shrink-0">
                  E
                </div>
                <div>
                  <h3 className="text-xl font-bold text-[#202124] group-hover:text-[#1a73e8] transition-colors">ECSS Standards</h3>
                  <p className="text-sm text-[#5f6368]">European Cooperation for Space Standardization</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mb-5">
                <span className="bg-[#fef3c7] text-[#92400e] text-xs font-medium px-2.5 py-1 rounded-full">M — Management</span>
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs font-medium px-2.5 py-1 rounded-full">E — Engineering</span>
                <span className="bg-[#d1fae5] text-[#065f46] text-xs font-medium px-2.5 py-1 rounded-full">Q — Quality</span>
                <span className="bg-[#ede9fe] text-[#4c1d95] text-xs font-medium px-2.5 py-1 rounded-full">+4 branches</span>
              </div>
              <div className="grid grid-cols-3 gap-3 mb-5 text-center">
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">7</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">Branches</p>
                </div>
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">139</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">Standards</p>
                </div>
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">25K+</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">Requirements</p>
                </div>
              </div>
              <p className="text-sm text-[#1a73e8] font-medium group-hover:underline">Explore ECSS Hub →</p>
            </Link>

            {/* NASA Hub */}
            <Link href="/nasa" className="group block bg-white border border-[#dadce0] rounded-2xl p-8 hover:shadow-lg hover:border-[#0b3d91] transition-all duration-200">
              <div className="flex items-center gap-4 mb-5">
                <div className="w-14 h-14 bg-[#0b3d91] rounded-xl flex items-center justify-center text-white font-bold text-xl flex-shrink-0">
                  N
                </div>
                <div>
                  <h3 className="text-xl font-bold text-[#202124] group-hover:text-[#0b3d91] transition-colors">NASA Standards</h3>
                  <p className="text-sm text-[#5f6368]">NASA Technical Standards Program (NTSP)</p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2 mb-5">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs font-medium px-2.5 py-1 rounded-full">Structures 5000</span>
                <span className="bg-[#fef3c7] text-[#92400e] text-xs font-medium px-2.5 py-1 rounded-full">Electrical 4000</span>
                <span className="bg-[#d1fae5] text-[#065f46] text-xs font-medium px-2.5 py-1 rounded-full">Safety 8000</span>
                <span className="bg-[#ede9fe] text-[#4c1d95] text-xs font-medium px-2.5 py-1 rounded-full">+4 domains</span>
              </div>
              <div className="grid grid-cols-3 gap-3 mb-5 text-center">
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">83</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">Documents</p>
                </div>
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">10</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">NASA Centers</p>
                </div>
                <div className="bg-[#f1f3f4] rounded-lg p-3">
                  <p className="text-2xl font-bold text-[#202124]">7</p>
                  <p className="text-xs text-[#5f6368] mt-0.5">Disciplines</p>
                </div>
              </div>
              <p className="text-sm text-[#0b3d91] font-medium group-hover:underline">Explore NASA Hub →</p>
            </Link>
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
              <span className="text-xl font-bold text-[#202124]">Larun<span className="text-[#1a73e8]">.</span><span className="text-[#1a73e8]">Space</span></span>
            </Link>

            {/* Links */}
            <div className="flex flex-wrap items-center gap-6 text-sm text-[#5f6368]">
              <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                Larun Engineering
              </a>
              <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                GitHub
              </a>
              <Link href="/ecss" className="hover:text-[#202124] transition-colors">
                ECSS Standards
              </Link>
              <Link href="/nasa" className="hover:text-[#202124] transition-colors">
                NASA Standards
              </Link>
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
