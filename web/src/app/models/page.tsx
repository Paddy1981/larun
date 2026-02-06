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

// Featured model - Multi-View Exoplanet Detection (production-ready)
const featuredModel: ModelCard = {
  id: 'multiview-exoplanet',
  name: 'MULTIVIEW-EXOPLANET',
  description: 'Advanced multi-view architecture for exoplanet detection. Analyzes global (2001 pts), local (201 pts), and secondary eclipse views simultaneously for robust transit identification.',
  accuracy: '74.2%',
  metric: 'auc',
  size: '327KB',
  downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/MULTIVIEW-EXOPLANET_weights.npz',
  trained: true,
};

const detectionModels: ModelCard[] = [
  {
    id: 'exoplanet-001',
    name: 'EXOPLANET-001',
    description: 'Primary exoplanet transit detection. Identifies periodic brightness dips characteristic of planetary transits using 1D CNN architecture.',
    accuracy: '80.2%',
    size: '11KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/EXOPLANET-001_weights.npz',
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
    description: 'Stellar flare detection. Identifies weak, moderate, strong, and superflare events in light curves.',
    accuracy: '84.2%',
    size: '3KB',
    downloadUrl: 'https://github.com/Paddy1981/larun/raw/main/models/trained/FLARE-001_weights.npz',
    trained: true,
  },
  {
    id: 'microlens-001',
    name: 'MICROLENS-001',
    description: 'Microlensing event detection. Identifies single lens, binary lens, and planetary microlensing events.',
    accuracy: '84.8%',
    size: '32KB',
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

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-10">
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">9</div>
            <div className="text-sm text-[#5f6368] mt-1">Trained Models</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">99.8%</div>
            <div className="text-sm text-[#5f6368] mt-1">Best Accuracy</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">&lt;35KB</div>
            <div className="text-sm text-[#5f6368] mt-1">Avg Model Size</div>
          </div>
          <div className="bg-white p-5 rounded-xl border border-[#dadce0] text-center">
            <div className="text-3xl font-bold text-[#202124]">&lt;50ms</div>
            <div className="text-sm text-[#5f6368] mt-1">Inference Time</div>
          </div>
        </div>

        {/* Featured Model */}
        <h2 className="text-2xl font-semibold text-[#202124] mb-4">Featured: Multi-View Architecture</h2>
        <div className="bg-gradient-to-r from-[#1a73e8] to-[#4285f4] text-white rounded-xl p-6 mb-10">
          <div className="flex flex-col md:flex-row md:items-center gap-6">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-xl font-bold font-mono">{featuredModel.name}</h3>
                <span className="px-2 py-0.5 bg-white/20 text-xs rounded-full">NEW</span>
              </div>
              <p className="text-blue-100 mb-4">{featuredModel.description}</p>
              <div className="flex gap-6 text-sm mb-4">
                <span>AUC: <strong>{featuredModel.accuracy}</strong></span>
                <span>Size: <strong>{featuredModel.size}</strong></span>
                <span>Format: <strong>.npz (NumPy)</strong></span>
              </div>
              <a
                href={featuredModel.downloadUrl}
                className="inline-block px-5 py-2.5 bg-white text-[#1a73e8] font-medium rounded-lg hover:bg-blue-50 transition-colors text-sm"
              >
                Download Weights
              </a>
            </div>
            <div className="hidden md:block text-right">
              <div className="text-6xl font-bold opacity-20">MV</div>
              <div className="text-sm opacity-60">Multi-View CNN</div>
            </div>
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
              {model.downloadUrl ? (
                <a
                  href={model.downloadUrl}
                  className="block text-center px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Download .npz
                </a>
              ) : (
                <span className="block text-center px-4 py-2 bg-[#f1f3f4] text-[#5f6368] text-sm rounded-lg">
                  Coming Soon
                </span>
              )}
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
              {model.downloadUrl ? (
                <a
                  href={model.downloadUrl}
                  className="block text-center px-4 py-2 bg-[#1a73e8] hover:bg-[#1557b0] text-white text-sm font-medium rounded-lg transition-colors"
                >
                  Download .npz
                </a>
              ) : (
                <span className="block text-center px-4 py-2 bg-[#f1f3f4] text-[#5f6368] text-sm rounded-lg">
                  Coming Soon
                </span>
              )}
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
