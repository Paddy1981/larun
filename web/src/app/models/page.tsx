import Link from 'next/link';

interface ModelCard {
  id: string;
  name: string;
  description: string;
  accuracy: string;
  size: string;
}

const detectionModels: ModelCard[] = [
  {
    id: 'exoplanet-001',
    name: 'EXOPLANET-001',
    description: 'Primary exoplanet transit detection. Identifies periodic brightness dips characteristic of planetary transits.',
    accuracy: '81.8%',
    size: '45KB',
  },
  {
    id: 'vstar-001',
    name: 'VSTAR-001',
    description: 'Variable star classification. Distinguishes intrinsic stellar variability from transit signals.',
    accuracy: '89.2%',
    size: '38KB',
  },
  {
    id: 'flare-001',
    name: 'FLARE-001',
    description: 'Stellar flare detection. Identifies and flags flare events that could mask or mimic transits.',
    accuracy: '94.1%',
    size: '32KB',
  },
];

const analysisModels: ModelCard[] = [
  {
    id: 'spectype-001',
    name: 'SPECTYPE-001',
    description: 'Stellar spectral classification. Estimates stellar type from photometric data.',
    accuracy: '87.5%',
    size: '52KB',
  },
  {
    id: 'astero-001',
    name: 'ASTERO-001',
    description: 'Asteroseismology analysis. Detects stellar oscillations to determine stellar properties.',
    accuracy: '76.3%',
    size: '48KB',
  },
  {
    id: 'galaxy-001',
    name: 'GALAXY-001',
    description: 'Galaxy morphology classification. Identifies galaxy types from imaging data.',
    accuracy: '91.7%',
    size: '67KB',
  },
];

export default function ModelsPage() {
  return (
    <div className="min-h-screen bg-[#f1f3f4]">
      {/* Top Navigation */}
      <nav className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-[#dadce0] flex items-center px-6 z-50">
        <div className="flex items-center gap-2">
          <Link href="/" className="text-[22px] font-medium text-[#202124] no-underline">
            Larun<span className="text-[#5f6368]">.</span>
          </Link>
          <span className="text-sm text-[#5f6368] font-normal pl-2 border-l border-[#dadce0] ml-2">AstroTinyML</span>
        </div>

        <div className="flex-1 flex justify-center gap-2">
          <Link href="/" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Home
          </Link>
          <Link href="/dashboard" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Dashboard
          </Link>
          <Link href="/#pricing" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Pricing
          </Link>
          <Link href="/guide" className="px-4 py-2 text-[#1a73e8] text-sm font-medium rounded no-underline">
            Docs
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/dashboard" className="px-5 py-2 bg-[#202124] text-white text-sm font-medium rounded no-underline hover:bg-[#3c4043] transition-colors">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-16 max-w-[1000px] mx-auto px-8 py-8">
        <h1 className="text-[32px] font-normal text-[#202124] mb-4">TinyML Models</h1>
        <p className="text-[#3c4043] mb-6">
          Larun uses a federation of specialized TinyML models, each optimized for a specific astronomical detection
          or analysis task. All models are under 100KB, enabling edge deployment and browser-based inference.
        </p>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-6">
          <div className="bg-white p-5 rounded-lg border border-[#dadce0] text-center">
            <div className="text-[28px] font-medium text-[#202124]">9</div>
            <div className="text-[13px] text-[#5f6368] mt-1">Specialized Models</div>
          </div>
          <div className="bg-white p-5 rounded-lg border border-[#dadce0] text-center">
            <div className="text-[28px] font-medium text-[#202124]">&lt;100KB</div>
            <div className="text-[13px] text-[#5f6368] mt-1">Per Model</div>
          </div>
          <div className="bg-white p-5 rounded-lg border border-[#dadce0] text-center">
            <div className="text-[28px] font-medium text-[#202124]">81.8%</div>
            <div className="text-[13px] text-[#5f6368] mt-1">Detection Accuracy</div>
          </div>
          <div className="bg-white p-5 rounded-lg border border-[#dadce0] text-center">
            <div className="text-[28px] font-medium text-[#202124]">&lt;50ms</div>
            <div className="text-[13px] text-[#5f6368] mt-1">Inference Time</div>
          </div>
        </div>

        {/* Detection Models */}
        <h2 className="text-2xl font-medium text-[#202124] mt-8 mb-4 pt-6 border-t border-[#dadce0]">
          Detection Models
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 my-6">
          {detectionModels.map((model) => (
            <div key={model.id} className="bg-white p-6 rounded-lg border border-[#dadce0]">
              <h4 className="text-base font-medium mb-2 font-mono text-[#1a73e8]">{model.name}</h4>
              <p className="text-sm text-[#3c4043] mb-3">{model.description}</p>
              <div className="flex gap-4 text-xs text-[#5f6368]">
                <span>Accuracy: {model.accuracy}</span>
                <span>Size: {model.size}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Analysis Models */}
        <h2 className="text-2xl font-medium text-[#202124] mt-8 mb-4 pt-6 border-t border-[#dadce0]">
          Analysis Models
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 my-6">
          {analysisModels.map((model) => (
            <div key={model.id} className="bg-white p-6 rounded-lg border border-[#dadce0]">
              <h4 className="text-base font-medium mb-2 font-mono text-[#1a73e8]">{model.name}</h4>
              <p className="text-sm text-[#3c4043] mb-3">{model.description}</p>
              <div className="flex gap-4 text-xs text-[#5f6368]">
                <span>Accuracy: {model.accuracy}</span>
                <span>Size: {model.size}</span>
              </div>
            </div>
          ))}
        </div>

        {/* Model Architecture */}
        <h2 className="text-2xl font-medium text-[#202124] mt-8 mb-4 pt-6 border-t border-[#dadce0]">
          Model Architecture
        </h2>
        <p className="text-[#3c4043] mb-4">
          All Larun models share a common architecture optimized for TinyML deployment:
        </p>
        <ul className="pl-6 text-[#3c4043] space-y-2">
          <li>
            <strong>Input:</strong> Normalized light curve (flux vs. time)
          </li>
          <li>
            <strong>Preprocessing:</strong> Detrending, outlier removal, normalization
          </li>
          <li>
            <strong>Feature Extraction:</strong> 1D Convolutional layers
          </li>
          <li>
            <strong>Classification:</strong> Dense layers with dropout
          </li>
          <li>
            <strong>Output:</strong> Class probabilities with confidence scores
          </li>
        </ul>

        {/* Additional Models Section */}
        <h2 className="text-2xl font-medium text-[#202124] mt-8 mb-4 pt-6 border-t border-[#dadce0]">
          Coming Soon
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5 my-6">
          <div className="bg-white p-6 rounded-lg border border-[#dadce0] border-dashed opacity-70">
            <h4 className="text-base font-medium mb-2 font-mono text-[#5f6368]">BINARY-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Eclipsing binary detection. Identifies binary star systems that can mimic planetary transits.</p>
            <div className="flex gap-4 text-xs text-[#9aa0a6]">
              <span>In Development</span>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg border border-[#dadce0] border-dashed opacity-70">
            <h4 className="text-base font-medium mb-2 font-mono text-[#5f6368]">MULTI-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Multi-planet detection. Identifies systems with multiple transiting planets.</p>
            <div className="flex gap-4 text-xs text-[#9aa0a6]">
              <span>In Development</span>
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg border border-[#dadce0] border-dashed opacity-70">
            <h4 className="text-base font-medium mb-2 font-mono text-[#5f6368]">HABIT-001</h4>
            <p className="text-sm text-[#5f6368] mb-3">Habitability scoring. Estimates potential for habitable conditions.</p>
            <div className="flex gap-4 text-xs text-[#9aa0a6]">
              <span>In Development</span>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 border-t border-[#dadce0] bg-white text-center">
        <div className="flex justify-center gap-6 mb-4">
          <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            laruneng.com
          </a>
          <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            GitHub
          </a>
          <Link href="/guide" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            Documentation
          </Link>
          <Link href="/#pricing" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            Pricing
          </Link>
        </div>
        <p className="text-xs text-[#5f6368]">&copy; {new Date().getFullYear()} Larun. AstroTinyML. All rights reserved.</p>
      </footer>
    </div>
  );
}
