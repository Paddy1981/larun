import Link from 'next/link';
import Header from '@/components/Header';

interface ModelCard {
  id: string;
  name: string;
  description: string;
  accuracy: string;
  size: string;
  downloadUrl?: string;
}

const detectionModels: ModelCard[] = [
  {
    id: 'exoplanet-001',
    name: 'EXOPLANET-001',
    description: 'Primary exoplanet transit detection. Identifies periodic brightness dips characteristic of planetary transits.',
    accuracy: '82%',
    size: '48KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/exoplanet-001.tflite',
  },
  {
    id: 'vstar-001',
    name: 'VSTAR-001',
    description: 'Variable star classification. Distinguishes intrinsic stellar variability from transit signals.',
    accuracy: '87%',
    size: '72KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/vstar-001.tflite',
  },
  {
    id: 'flare-001',
    name: 'FLARE-001',
    description: 'Stellar flare detection. Identifies and flags flare events that could mask or mimic transits.',
    accuracy: '91%',
    size: '32KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/flare-001.tflite',
  },
  {
    id: 'supernova-001',
    name: 'SUPERNOVA-001',
    description: 'Supernova and transient detection. Classifies explosive stellar events including Type Ia, II, and kilonovae.',
    accuracy: '86%',
    size: '80KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/supernova-001.tflite',
  },
  {
    id: 'microlens-001',
    name: 'MICROLENS-001',
    description: 'Gravitational microlensing detection. Identifies single, binary, and planetary lensing events.',
    accuracy: '84%',
    size: '72KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/microlens-001.tflite',
  },
];

const analysisModels: ModelCard[] = [
  {
    id: 'spectype-001',
    name: 'SPECTYPE-001',
    description: 'Stellar spectral classification. Estimates stellar type (O, B, A, F, G, K, M, L) from photometric data.',
    accuracy: '85%',
    size: '40KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/spectype-001.tflite',
  },
  {
    id: 'astero-001',
    name: 'ASTERO-001',
    description: 'Asteroseismology analysis. Detects stellar oscillations to determine stellar properties.',
    accuracy: '83%',
    size: '60KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/astero-001.tflite',
  },
  {
    id: 'galaxy-001',
    name: 'GALAXY-001',
    description: 'Galaxy morphology classification. Identifies elliptical, spiral, barred spiral, and irregular galaxies.',
    accuracy: '79%',
    size: '88KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/releases/download/v0.1.0/galaxy-001.tflite',
  },
];

export default function ModelsPage() {
  return (
    <div className="min-h-screen bg-white">
      <Header />

      {/* Main Content */}
      <main className="pt-24 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        <h1 className="text-3xl font-bold text-[#202124] mb-3">TinyML Models</h1>
        <p className="text-[#5f6368] mb-8 max-w-3xl">
          Larun uses a federation of specialized TinyML models, each optimized for a specific astronomical detection
          or analysis task. All models are under 100KB, enabling edge deployment and browser-based inference.
        </p>

        {/* Download Section */}
        <div className="bg-[#1a73e8] text-white rounded-xl p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold mb-2">Download All Models</h2>
              <p className="text-blue-100 text-sm">Get the complete model bundle for local inference on your own hardware.</p>
            </div>
            <div className="flex gap-3">
              <a
                href="https://github.com/Paddy1981/larun/releases/latest"
                target="_blank"
                rel="noopener noreferrer"
                className="px-5 py-2.5 bg-white text-[#1a73e8] font-medium rounded-lg hover:bg-blue-50 transition-colors text-sm"
              >
                Download Bundle
              </a>
              <a
                href="https://github.com/Paddy1981/larun"
                target="_blank"
                rel="noopener noreferrer"
                className="px-5 py-2.5 bg-transparent border border-white text-white font-medium rounded-lg hover:bg-white/10 transition-colors text-sm"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>

        {/* Subscription Info */}
        <div className="bg-[#fef7e0] border border-[#f9e79f] rounded-xl p-4 mb-8">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-[#b7950b] mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm text-[#7d6608] font-medium">All models are available to all users</p>
              <p className="text-xs text-[#9a7d0a] mt-1">
                Subscription tiers limit the number of analyses per month: Free (5), Monthly (50), Annual (Unlimited).
                <Link href="/settings/subscription" className="ml-1 underline hover:no-underline">View plans</Link>
              </p>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">8</div>
            <div className="text-sm text-[#5f6368] mt-1">Specialized Models</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">&lt;100KB</div>
            <div className="text-sm text-[#5f6368] mt-1">Per Model</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">84.7%</div>
            <div className="text-sm text-[#5f6368] mt-1">Avg. Accuracy</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">&lt;20ms</div>
            <div className="text-sm text-[#5f6368] mt-1">Avg. Inference</div>
          </div>
        </div>

        {/* Detection Models */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Detection Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
          {detectionModels.map((model) => (
            <div key={model.id} className="bg-white p-6 rounded-xl border border-[#dadce0] hover:shadow-md transition-shadow">
              <h4 className="text-base font-semibold mb-2 font-mono text-[#1a73e8]">{model.name}</h4>
              <p className="text-sm text-[#5f6368] mb-4">{model.description}</p>
              <div className="flex justify-between items-center text-xs text-[#5f6368] mb-4">
                <span>Accuracy: <strong className="text-[#202124]">{model.accuracy}</strong></span>
                <span>Size: <strong className="text-[#202124]">{model.size}</strong></span>
              </div>
              {model.downloadUrl && (
                <a
                  href={model.downloadUrl}
                  className="block text-center px-4 py-2 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] text-sm font-medium rounded-lg transition-colors"
                >
                  Download .tflite
                </a>
              )}
            </div>
          ))}
        </div>

        {/* Analysis Models */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Analysis Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
          {analysisModels.map((model) => (
            <div key={model.id} className="bg-white p-6 rounded-xl border border-[#dadce0] hover:shadow-md transition-shadow">
              <h4 className="text-base font-semibold mb-2 font-mono text-[#1a73e8]">{model.name}</h4>
              <p className="text-sm text-[#5f6368] mb-4">{model.description}</p>
              <div className="flex justify-between items-center text-xs text-[#5f6368] mb-4">
                <span>Accuracy: <strong className="text-[#202124]">{model.accuracy}</strong></span>
                <span>Size: <strong className="text-[#202124]">{model.size}</strong></span>
              </div>
              {model.downloadUrl && (
                <a
                  href={model.downloadUrl}
                  className="block text-center px-4 py-2 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] text-sm font-medium rounded-lg transition-colors"
                >
                  Download .tflite
                </a>
              )}
            </div>
          ))}
        </div>

        {/* Local Usage Section */}
        <div className="bg-[#f8f9fa] rounded-xl p-6 mb-10">
          <h2 className="text-xl font-semibold text-[#202124] mb-4">Run Models Locally</h2>
          <p className="text-[#5f6368] mb-4">
            Use our TinyML models on your own hardware with Python and TensorFlow Lite:
          </p>
          <pre className="bg-[#202124] text-green-400 p-4 rounded-lg text-sm overflow-x-auto mb-4">
{`# Install dependencies
pip install larun tflite-runtime numpy

# Load and run a model
from larun import ExoplanetDetector

detector = ExoplanetDetector()
result = detector.analyze("path/to/lightcurve.csv")

print(f"Detection: {result.detection}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Period: {result.period_days:.4f} days")`}
          </pre>
          <div className="flex gap-3">
            <a
              href="https://pypi.org/project/larun/"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
            >
              Install from PyPI
            </a>
            <Link
              href="/guide"
              className="px-4 py-2 bg-white hover:bg-[#f1f3f4] text-[#202124] text-sm font-medium rounded-lg border border-[#dadce0] transition-colors"
            >
              View Documentation
            </Link>
          </div>
        </div>

        {/* Model Architecture */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Model Architecture</h2>
        <p className="text-[#5f6368] mb-4">
          All Larun models share a common architecture optimized for TinyML deployment:
        </p>
        <div className="bg-white rounded-xl border border-[#dadce0] p-6 mb-10">
          <div className="grid md:grid-cols-5 gap-4 text-center">
            <div className="p-4">
              <div className="w-12 h-12 bg-[#1a73e8] text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">1</div>
              <p className="font-medium text-[#202124]">Input</p>
              <p className="text-xs text-[#5f6368]">Light curve data</p>
            </div>
            <div className="p-4">
              <div className="w-12 h-12 bg-[#1a73e8] text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">2</div>
              <p className="font-medium text-[#202124]">Preprocessing</p>
              <p className="text-xs text-[#5f6368]">Detrend & normalize</p>
            </div>
            <div className="p-4">
              <div className="w-12 h-12 bg-[#1a73e8] text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">3</div>
              <p className="font-medium text-[#202124]">Feature Extraction</p>
              <p className="text-xs text-[#5f6368]">1D Conv layers</p>
            </div>
            <div className="p-4">
              <div className="w-12 h-12 bg-[#1a73e8] text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">4</div>
              <p className="font-medium text-[#202124]">Classification</p>
              <p className="text-xs text-[#5f6368]">Dense + dropout</p>
            </div>
            <div className="p-4">
              <div className="w-12 h-12 bg-[#1a73e8] text-white rounded-full flex items-center justify-center mx-auto mb-2 font-bold">5</div>
              <p className="font-medium text-[#202124]">Output</p>
              <p className="text-xs text-[#5f6368]">Probabilities</p>
            </div>
          </div>
        </div>

        {/* Coming Soon */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Coming Soon</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div className="bg-white p-6 rounded-xl border-2 border-dashed border-[#dadce0] opacity-70">
            <h4 className="text-base font-semibold mb-2 font-mono text-[#5f6368]">BINARY-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Eclipsing binary detection. Identifies binary star systems that can mimic planetary transits.</p>
            <span className="inline-block px-3 py-1 bg-[#f1f3f4] text-xs text-[#5f6368] rounded-full">In Development</span>
          </div>
          <div className="bg-white p-6 rounded-xl border-2 border-dashed border-[#dadce0] opacity-70">
            <h4 className="text-base font-semibold mb-2 font-mono text-[#5f6368]">MULTI-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Multi-planet detection. Identifies systems with multiple transiting planets.</p>
            <span className="inline-block px-3 py-1 bg-[#f1f3f4] text-xs text-[#5f6368] rounded-full">In Development</span>
          </div>
          <div className="bg-white p-6 rounded-xl border-2 border-dashed border-[#dadce0] opacity-70">
            <h4 className="text-base font-semibold mb-2 font-mono text-[#5f6368]">HABIT-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Habitability scoring. Estimates potential for habitable conditions.</p>
            <span className="inline-block px-3 py-1 bg-[#f1f3f4] text-xs text-[#5f6368] rounded-full">In Development</span>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="flex justify-center gap-6 mb-4">
            <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm hover:text-[#202124]">
              laruneng.com
            </a>
            <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm hover:text-[#202124]">
              GitHub
            </a>
            <Link href="/guide" className="text-[#5f6368] text-sm hover:text-[#202124]">
              Documentation
            </Link>
          </div>
          <p className="text-xs text-[#5f6368]">&copy; {new Date().getFullYear()} Larun Engineering. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
