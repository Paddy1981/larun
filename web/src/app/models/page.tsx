import Link from 'next/link';
import Header from '@/components/Header';

interface ModelCard {
  id: string;
  name: string;
  description: string;
  accuracy: string;
  size: string;
  downloadUrl?: string;
  trained?: boolean;
  metric?: string; // 'accuracy' or 'auc'
}

// Removed MULTIVIEW-EXOPLANET - superseded by EXOPLANET-001 (98% accuracy vs 74.2% AUC, 43KB vs 327KB)

const detectionModels: ModelCard[] = [
  {
    id: 'exoplanet-001',
    name: 'EXOPLANET-001',
    description: 'Primary exoplanet transit detection trained on real Kepler/TESS light curves. Feature-based classifier distinguishing transits, eclipsing binaries, and noise.',
    accuracy: '98.0%',
    size: '43KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/EXOPLANET-001_real_weights.npz',
    trained: true,
  },
  {
    id: 'vstar-001',
    name: 'VSTAR-001',
    description: 'Variable star classification. Distinguishes Cepheids, RR Lyrae, Delta Scuti, and other variable star types.',
    accuracy: '99.8%',
    size: '27KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/VSTAR-001_weights.npz',
    trained: true,
  },
  {
    id: 'flare-001',
    name: 'FLARE-001',
    description: 'Stellar flare detection. Feature-based classifier for quiescent, flare, and strong flare states.',
    accuracy: '96.7%',
    size: '5KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/FLARE-001_weights.npz',
    trained: true,
  },
  {
    id: 'microlens-001',
    name: 'MICROLENS-001',
    description: 'Microlensing event detection. Feature-based classifier for no event, simple lens, and complex events.',
    accuracy: '99.4%',
    size: '5KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/MICROLENS-001_weights.npz',
    trained: true,
  },
  {
    id: 'supernova-001',
    name: 'SUPERNOVA-001',
    description: 'Supernova and transient detection. Feature-based classifier for Type I, II supernovae, kilonovae, and TDEs.',
    accuracy: '100.0%',
    size: '3KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/SUPERNOVA-001_weights.npz',
    trained: true,
  },
];

const discoveryModels: ModelCard[] = [
  {
    id: 'vardet-001',
    name: 'VARDET-001',
    description: 'VARnet-inspired variability detector. Lomb-Scargle + Daubechies wavelet features fed into a Random Forest. Classifies NON_VARIABLE, TRANSIENT, PULSATOR, ECLIPSING — the same approach that found 1.5M objects in NEOWISE.',
    accuracy: '97.2%',
    size: '~50KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/VARDET-001_weights.npz',
    trained: true,
  },
  {
    id: 'anomaly-001',
    name: 'ANOMALY-001',
    description: 'Isolation Forest anomaly detector on 14-dim feature vectors. Flags objects with unusual variability patterns — designed to catch Boyajian\'s Star analogs and other unexplained light curves.',
    accuracy: '—',
    size: '~20KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/ANOMALY-001_weights.npz',
    trained: true,
    metric: 'unsupervised',
  },
  {
    id: 'deblend-001',
    name: 'DEBLEND-001',
    description: 'Detects blended / contaminated TESS pixels using multi-frequency analysis and pixel-crowding metrics (CROWDSAP, FLFRCSAP). Flags CLEAN, MILD_BLEND, STRONG_BLEND, CONTAMINATED.',
    accuracy: '94.1%',
    size: '~15KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/DEBLEND-001_weights.npz',
    trained: true,
  },
  {
    id: 'periodogram-001',
    name: 'PERIODOGRAM-001',
    description: 'Consensus period finder using 4 methods: Lomb-Scargle, BLS, Phase Dispersion Minimization, and Autocorrelation. Returns best period, confidence, and type (transit / pulsation / rotation / irregular).',
    accuracy: '—',
    size: '~5KB',
    downloadUrl: undefined,
    trained: true,
    metric: 'algorithmic',
  },
];

const analysisModels: ModelCard[] = [
  {
    id: 'spectype-001',
    name: 'SPECTYPE-001',
    description: 'Stellar spectral classification. Classifies O, B, A, F, G, K, M, L type stars from photometric features.',
    accuracy: '95.0%',
    size: '5KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/SPECTYPE-001_weights.npz',
    trained: true,
  },
  {
    id: 'astero-001',
    name: 'ASTERO-001',
    description: 'Asteroseismology analysis. Detects solar-like, red giant, delta Scuti, and gamma Dor oscillations.',
    accuracy: '99.8%',
    size: '21KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/ASTERO-001_weights.npz',
    trained: true,
  },
  {
    id: 'galaxy-001',
    name: 'GALAXY-001',
    description: 'Galaxy morphology classification. Feature-based classifier for elliptical, spiral, barred spiral, irregular, and merger galaxies.',
    accuracy: '99.9%',
    size: '4KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/GALAXY-001_weights.npz',
    trained: true,
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

        {/* Try Cloud Platform CTA */}
        <div className="bg-gradient-to-r from-[#1a73e8] to-[#174ea6] text-white rounded-xl p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <div className="inline-flex items-center gap-2 bg-white/20 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-medium mb-3">
                <span>✨</span>
                <span>NEW: Cloud Platform</span>
              </div>
              <h2 className="text-xl font-semibold mb-2">Try These Models Online</h2>
              <p className="text-white/90 text-sm mb-2">Upload FITS files and run instant inference with any of these 8 models in the cloud.</p>
              <p className="text-white/75 text-xs">5 free analyses per month • No setup required • Results in &lt;100ms</p>
            </div>
            <div className="flex gap-3">
              <Link
                href="/cloud"
                className="px-5 py-2.5 bg-white text-[#1a73e8] font-medium rounded-lg hover:bg-blue-50 transition-colors text-sm whitespace-nowrap"
              >
                Try Cloud Platform →
              </Link>
              <Link
                href="/cloud/pricing"
                className="px-5 py-2.5 bg-transparent border border-white text-white font-medium rounded-lg hover:bg-white/10 transition-colors text-sm whitespace-nowrap"
              >
                View Pricing
              </Link>
            </div>
          </div>
        </div>

        {/* Download Section */}
        <div className="bg-white border-2 border-[#dadce0] rounded-xl p-6 mb-8">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
            <div>
              <h2 className="text-xl font-semibold text-[#202124] mb-2">Download Models for Local Use</h2>
              <p className="text-[#5f6368] text-sm">Get the complete model bundle for local inference on your own hardware (free forever).</p>
            </div>
            <div className="flex gap-3">
              <a
                href="https://github.com/Paddy1981/larun/releases/latest"
                target="_blank"
                rel="noopener noreferrer"
                className="px-5 py-2.5 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors text-sm"
              >
                Download Bundle
              </a>
              <a
                href="https://github.com/Paddy1981/larun"
                target="_blank"
                rel="noopener noreferrer"
                className="px-5 py-2.5 bg-white hover:bg-[#f1f3f4] text-[#202124] font-medium rounded-lg border border-[#dadce0] transition-colors text-sm"
              >
                View on GitHub
              </a>
            </div>
          </div>
        </div>

        {/* Usage Options Info */}
        <div className="bg-[#e8f5e9] border border-[#a5d6a7] rounded-xl p-4 mb-8">
          <div className="flex items-start gap-3">
            <svg className="w-5 h-5 text-[#2e7d32] mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"/>
            </svg>
            <div>
              <p className="text-sm text-[#1b5e20] font-medium">Two Ways to Use These Models</p>
              <p className="text-xs text-[#2e7d32] mt-1">
                <strong>Cloud Platform:</strong> Upload FITS files for instant inference (5 free/month, then $9/month for 50 analyses).
                <Link href="/cloud/pricing" className="ml-1 underline hover:no-underline">View plans</Link>
              </p>
              <p className="text-xs text-[#2e7d32] mt-1">
                <strong>Local Download:</strong> Download models and run unlimited inference on your own hardware (free forever).
                <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="ml-1 underline hover:no-underline">Get started</a>
              </p>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">12</div>
            <div className="text-sm text-[#5f6368] mt-1">Trained Models</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#1a73e8] text-center">
            <div className="text-3xl font-bold text-[#1a73e8]">98.0%</div>
            <div className="text-sm text-[#5f6368] mt-1">Real Data Accuracy</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">&lt;35KB</div>
            <div className="text-sm text-[#5f6368] mt-1">Avg Model Size</div>
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
            <div key={model.id} className={`bg-white p-6 rounded-xl border ${model.trained ? 'border-[#1a73e8]' : 'border-[#dadce0]'} hover:shadow-md transition-shadow`}>
              <div className="flex items-center gap-2 mb-2">
                <h4 className="text-base font-semibold font-mono text-[#1a73e8]">{model.name}</h4>
                {model.trained && <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">Trained</span>}
                {!model.trained && <span className="px-2 py-0.5 bg-gray-100 text-gray-500 text-xs rounded-full">Target</span>}
              </div>
              <p className="text-sm text-[#5f6368] mb-4">{model.description}</p>
              <div className="flex justify-between items-center text-xs text-[#5f6368] mb-4">
                <span>Accuracy: <strong className="text-[#202124]">{model.accuracy}</strong></span>
                <span>Size: <strong className="text-[#202124]">{model.size}</strong></span>
              </div>
              <div className="flex gap-2">
                <Link
                  href="/cloud/analyze"
                  className="flex-1 text-center px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Try on Cloud
                </Link>
                {model.downloadUrl ? (
                  <a
                    href={model.downloadUrl}
                    className="flex-1 text-center px-4 py-2 bg-white hover:bg-[#f1f3f4] text-[#202124] text-sm font-medium rounded-lg border border-[#dadce0] transition-colors"
                  >
                    Download
                  </a>
                ) : (
                  <span className="flex-1 text-center px-4 py-2 bg-[#f1f3f4] text-[#5f6368] text-sm rounded-lg">
                    Soon
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Discovery Federation Models */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-semibold text-[#202124]">Discovery Federation Models</h2>
          <Link href="/discover" className="text-sm text-[#1a73e8] hover:underline font-medium">
            Try Citizen Discovery Engine →
          </Link>
        </div>
        <p className="text-sm text-[#5f6368] mb-5">
          Layer-2 server models running in the Citizen Discovery Engine. Inspired by Matteo Paz&apos;s VARnet
          which found 1.5M objects in NEOWISE — these models form the core of <Link href="/discover" className="text-[#1a73e8] hover:underline">larun.space/discover</Link>.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-10">
          {discoveryModels.map((model) => (
            <div key={model.id} className="bg-white p-6 rounded-xl border border-[#7c3aed] hover:shadow-md transition-shadow">
              <div className="flex items-center gap-2 mb-2">
                <h4 className="text-base font-semibold font-mono text-[#7c3aed]">{model.name}</h4>
                <span className="px-2 py-0.5 bg-purple-100 text-purple-700 text-xs rounded-full">
                  {(model as ModelCard & { metric?: string }).metric === 'algorithmic'
                    ? 'Algorithmic'
                    : (model as ModelCard & { metric?: string }).metric === 'unsupervised'
                    ? 'Unsupervised'
                    : 'Trained'}
                </span>
              </div>
              <p className="text-sm text-[#5f6368] mb-4">{model.description}</p>
              <div className="flex justify-between items-center text-xs text-[#5f6368] mb-4">
                <span>Accuracy: <strong className="text-[#202124]">{model.accuracy}</strong></span>
                <span>Size: <strong className="text-[#202124]">{model.size}</strong></span>
              </div>
              <div className="flex gap-2">
                <Link
                  href="/discover"
                  className="flex-1 text-center px-4 py-2 bg-[#7c3aed] hover:bg-[#6d28d9] text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Try in Discovery
                </Link>
                {model.downloadUrl ? (
                  <a
                    href={model.downloadUrl}
                    className="flex-1 text-center px-4 py-2 bg-white hover:bg-[#f1f3f4] text-[#202124] text-sm font-medium rounded-lg border border-[#dadce0] transition-colors"
                  >
                    Download
                  </a>
                ) : (
                  <span className="flex-1 text-center px-4 py-2 bg-[#f1f3f4] text-[#5f6368] text-sm rounded-lg">
                    No weights needed
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Analysis Models */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Analysis Models</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
          {analysisModels.map((model) => (
            <div key={model.id} className={`bg-white p-6 rounded-xl border ${model.trained ? 'border-[#1a73e8]' : 'border-[#dadce0]'} hover:shadow-md transition-shadow`}>
              <div className="flex items-center gap-2 mb-2">
                <h4 className="text-base font-semibold font-mono text-[#1a73e8]">{model.name}</h4>
                {model.trained && <span className="px-2 py-0.5 bg-green-100 text-green-700 text-xs rounded-full">Trained</span>}
                {!model.trained && <span className="px-2 py-0.5 bg-gray-100 text-gray-500 text-xs rounded-full">Target</span>}
              </div>
              <p className="text-sm text-[#5f6368] mb-4">{model.description}</p>
              <div className="flex justify-between items-center text-xs text-[#5f6368] mb-4">
                <span>Accuracy: <strong className="text-[#202124]">{model.accuracy}</strong></span>
                <span>Size: <strong className="text-[#202124]">{model.size}</strong></span>
              </div>
              <div className="flex gap-2">
                <Link
                  href="/cloud/analyze"
                  className="flex-1 text-center px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Try on Cloud
                </Link>
                {model.downloadUrl ? (
                  <a
                    href={model.downloadUrl}
                    className="flex-1 text-center px-4 py-2 bg-white hover:bg-[#f1f3f4] text-[#202124] text-sm font-medium rounded-lg border border-[#dadce0] transition-colors"
                  >
                    Download
                  </a>
                ) : (
                  <span className="flex-1 text-center px-4 py-2 bg-[#f1f3f4] text-[#5f6368] text-sm rounded-lg">
                    Soon
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Local Usage Section */}
        <div className="bg-[#f8f9fa] rounded-xl p-6 mb-10">
          <h2 className="text-xl font-semibold text-[#202124] mb-4">Run Models Locally</h2>
          <p className="text-[#5f6368] mb-4">
            Use our TinyML models on your own hardware with pure NumPy - no TensorFlow required:
          </p>
          <pre className="bg-[#202124] text-green-400 p-4 rounded-lg text-sm overflow-x-auto mb-4">
{`# Clone the repository
git clone https://github.com/Paddy1981/larun.git
cd larun && pip install numpy

# Load and run a model
from src.model import get_model

# Basic detection
detector = get_model("EXOPLANET-001")
detector.load("models/trained/EXOPLANET-001_weights.npz")
predictions, confidence = detector.predict(lightcurve_data)

# Multi-view detection (advanced)
from src.model.multiview_exoplanet import MultiViewExoplanetDetector
mv_detector = MultiViewExoplanetDetector()
mv_detector.load("models/trained/MULTIVIEW-EXOPLANET_weights.npz")
probs = mv_detector.forward(global_view, local_view, secondary_view)`}
          </pre>
          <div className="flex gap-3">
            <a
              href="https://github.com/Paddy1981/larun"
              target="_blank"
              rel="noopener noreferrer"
              className="px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
            >
              View on GitHub
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
