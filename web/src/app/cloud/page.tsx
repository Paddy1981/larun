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
          <Link href="/pricing" className="bg-larun-black text-white px-8 py-4 rounded-lg hover:bg-gray-800">
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
          <h2 className="text-4xl text-center mb-16">Features</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-xl font-medium mb-4">Exoplanet Detection</h3>
              <p className="text-larun-medium-gray">Detect planetary transits in light curves</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-xl font-medium mb-4">Variable Stars</h3>
              <p className="text-larun-medium-gray">Classify stellar variability</p>
            </div>
            <div className="bg-white p-6 rounded-lg">
              <h3 className="text-xl font-medium mb-4">More Models</h3>
              <p className="text-larun-medium-gray">6 additional specialized models</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
