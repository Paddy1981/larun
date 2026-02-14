/**
 * LARUN.SPACE Cloud - Simple Landing Page (Testing)
 */

import Link from 'next/link'

export default function Home() {
  return (
    <div className="pt-16">
      <section className="max-w-7xl mx-auto px-6 py-24 text-center">
        <h1 className="text-6xl font-medium mb-6">
          <span className="text-larun-black">LARUN</span>
          <span className="text-larun-medium-gray">.</span>
          <span className="text-larun-medium-gray">SPACE</span>
        </h1>

        <p className="text-2xl text-larun-medium-gray mb-4 max-w-3xl mx-auto">
          Analyze Astronomical Data with TinyML
        </p>

        <p className="text-lg text-larun-medium-gray mb-12 max-w-2xl mx-auto">
          Upload FITS files, select from 8 specialized models, and get instant classifications.
        </p>

        <div className="flex gap-4 justify-center mb-16">
          <Link href="/cloud/auth/signup" className="bg-larun-black text-white px-8 py-4 rounded-lg hover:bg-gray-800 transition-colors">
            Start Free Trial
          </Link>
          <Link href="/cloud/pricing" className="bg-white text-larun-black px-8 py-4 rounded-lg border-2 border-larun-black hover:bg-gray-50 transition-colors">
            View Pricing
          </Link>
        </div>

        <div className="grid grid-cols-4 gap-8 max-w-4xl mx-auto">
          <div>
            <div className="text-3xl font-medium text-larun-black">8</div>
            <div className="text-sm text-larun-medium-gray">TinyML Models</div>
          </div>
          <div>
            <div className="text-3xl font-medium text-larun-black">91.4%</div>
            <div className="text-sm text-larun-medium-gray">Avg Accuracy</div>
          </div>
          <div>
            <div className="text-3xl font-medium text-larun-black">&lt;100KB</div>
            <div className="text-sm text-larun-medium-gray">Model Size</div>
          </div>
          <div>
            <div className="text-3xl font-medium text-larun-black">&lt;10ms</div>
            <div className="text-sm text-larun-medium-gray">Inference Time</div>
          </div>
        </div>
      </section>

      <section className="bg-larun-lighter-gray py-24">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl text-center mb-4">8 Specialized TinyML Models</h2>
          <p className="text-center text-larun-medium-gray mb-16">
            Each model optimized for a specific astronomical detection task
          </p>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">EXOPLANET-001</h3>
              <p className="text-sm text-larun-medium-gray">Transit detection</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">VSTAR-001</h3>
              <p className="text-sm text-larun-medium-gray">Variable star classification</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">FLARE-001</h3>
              <p className="text-sm text-larun-medium-gray">Stellar flare detection</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">MICROLENS-001</h3>
              <p className="text-sm text-larun-medium-gray">Microlensing events</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">SUPERNOVA-001</h3>
              <p className="text-sm text-larun-medium-gray">Transient detection</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">SPECTYPE-001</h3>
              <p className="text-sm text-larun-medium-gray">Spectral classification</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">ASTERO-001</h3>
              <p className="text-sm text-larun-medium-gray">Asteroseismology</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-lg font-medium mb-2">GALAXY-001</h3>
              <p className="text-sm text-larun-medium-gray">Galaxy morphology</p>
            </div>
          </div>

          <div className="text-center">
            <Link href="/models" className="inline-block text-larun-black hover:underline">
              View detailed model specifications →
            </Link>
          </div>
        </div>
      </section>

      {/* Quick Start Section */}
      <section className="py-24 bg-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl mb-6">Get Started in Minutes</h2>
          <p className="text-lg text-larun-medium-gray mb-12">
            No setup, no installations. Just upload and analyze.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            <div>
              <div className="w-12 h-12 bg-larun-black text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                1
              </div>
              <h3 className="font-medium mb-2">Sign Up Free</h3>
              <p className="text-sm text-larun-medium-gray">Create account in seconds</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-larun-black text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                2
              </div>
              <h3 className="font-medium mb-2">Upload FITS File</h3>
              <p className="text-sm text-larun-medium-gray">Drag & drop your data</p>
            </div>
            <div>
              <div className="w-12 h-12 bg-larun-black text-white rounded-full flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                3
              </div>
              <h3 className="font-medium mb-2">Get Results</h3>
              <p className="text-sm text-larun-medium-gray">Instant classification</p>
            </div>
          </div>

          <Link href="/cloud/auth/signup" className="inline-block bg-larun-black text-white px-8 py-4 rounded-lg hover:bg-gray-800 transition-colors">
            Start Analyzing Now →
          </Link>
        </div>
      </section>
    </div>
  )
}
