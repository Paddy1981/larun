'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface NavItem {
  id: string;
  label: string;
}

interface NavSection {
  title: string;
  items: NavItem[];
}

const navSections: NavSection[] = [
  {
    title: 'User Guide',
    items: [
      { id: 'welcome', label: 'Welcome' },
      { id: 'first-steps', label: 'First Steps' },
      { id: 'understanding-results', label: 'Understanding Results' },
    ],
  },
  {
    title: 'Workflows',
    items: [
      { id: 'transit-search', label: 'Transit Search' },
      { id: 'candidate-vetting', label: 'Candidate Vetting' },
      { id: 'habitability', label: 'Habitability Analysis' },
    ],
  },
  {
    title: 'Resources',
    items: [
      { id: 'glossary', label: 'Glossary' },
      { id: 'references', label: 'References' },
    ],
  },
];

export default function GuidePage() {
  const [activeSection, setActiveSection] = useState('welcome');

  useEffect(() => {
    const handleScroll = () => {
      const sections = document.querySelectorAll('section[id]');
      let currentSection = 'welcome';

      sections.forEach((section) => {
        const sectionTop = section.getBoundingClientRect().top;
        if (sectionTop < 150) {
          currentSection = section.id;
        }
      });

      setActiveSection(currentSection);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-[#f1f3f4]">
      {/* Top Navigation */}
      <nav className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-[#dadce0] flex items-center px-6 z-50">
        <div className="flex items-center gap-2">
          <Link href="/" className="text-[22px] font-medium text-[#202124] no-underline">
            Larun<span className="text-[#1a73e8]">.</span><span className="text-[#1a73e8]">Space</span>
          </Link>
        </div>

        <div className="flex-1 flex justify-center gap-2">
          <Link href="/" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Home
          </Link>
          <Link href="/models" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Models
          </Link>
          <Link href="/cloud" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Cloud
          </Link>
          <Link href="/cloud/pricing" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Pricing
          </Link>
          <Link href="/guide" className="px-4 py-2 text-[#1a73e8] text-sm font-medium rounded no-underline">
            Docs
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/cloud/auth/signup" className="px-5 py-2 bg-[#202124] text-white text-sm font-medium rounded no-underline hover:bg-[#3c4043] transition-colors">
            Get Started
          </Link>
        </div>
      </nav>

      <div className="pt-16 flex">
        {/* Sidebar */}
        <aside className="fixed left-0 top-16 bottom-0 w-[260px] bg-white border-r border-[#dadce0] overflow-y-auto p-6">
          {navSections.map((section, idx) => (
            <div key={idx} className={idx > 0 ? 'mt-6' : ''}>
              <h3 className="text-xs font-medium text-[#5f6368] uppercase tracking-wider mb-3">
                {section.title}
              </h3>
              <ul className="list-none p-0 m-0 space-y-1">
                {section.items.map((item) => (
                  <li key={item.id}>
                    <button
                      onClick={() => scrollToSection(item.id)}
                      className={`w-full text-left px-3 py-2 text-sm rounded border-none cursor-pointer transition-colors ${
                        activeSection === item.id
                          ? 'bg-[#e8f0fe] text-[#1a73e8] font-medium'
                          : 'bg-transparent text-[#3c4043] hover:bg-[#f1f3f4]'
                      }`}
                    >
                      {item.label}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}

          <div className="mt-8 pt-6 border-t border-[#dadce0]">
            <Link href="/faq" className="flex items-center gap-2 text-sm text-[#5f6368] no-underline hover:text-[#202124]">
              <span>→</span> FAQ
            </Link>
            <a
              href="https://github.com/Paddy1981/larun"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-sm text-[#5f6368] no-underline hover:text-[#202124] mt-2"
            >
              <span>→</span> GitHub
            </a>
          </div>
        </aside>

        {/* Main Content */}
        <main className="ml-[260px] flex-1 p-8 max-w-[900px]">
          {/* Welcome Section */}
          <section id="welcome" className="mb-12">
            <h1 className="text-[32px] font-normal text-[#202124] mb-4">Welcome to LARUN.SPACE</h1>
            <p className="text-[#3c4043] text-base leading-relaxed mb-6">
              LARUN.SPACE is a TinyML-powered cloud platform for astronomical data analysis. With 8 specialized models,
              you can detect exoplanets, classify variable stars, identify stellar flares, find microlensing events,
              spot supernovae, classify spectra, perform asteroseismology, and analyze galaxy morphology.
            </p>

            <div className="bg-gradient-to-r from-[#1a73e8] to-[#174ea6] rounded-lg p-6 mb-6 text-white">
              <h3 className="text-lg font-semibold mb-3">☁️ Cloud Platform Now Available</h3>
              <p className="text-sm mb-4 opacity-90">
                Upload FITS files and run TinyML inference instantly with our cloud platform. No setup required.
              </p>
              <Link
                href="/cloud/auth/signup"
                className="inline-block bg-white text-[#1a73e8] px-6 py-2 rounded font-medium hover:bg-gray-100 transition-colors"
              >
                Start Free Trial
              </Link>
            </div>

            <div className="bg-white border border-[#dadce0] rounded-lg p-6 mb-6">
              <h3 className="text-lg font-medium text-[#202124] mb-4">8 TinyML Models Available</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">EXOPLANET-001</div>
                  <div className="text-xs text-[#5f6368]">Transit Detection</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">VSTAR-001</div>
                  <div className="text-xs text-[#5f6368]">Variable Stars</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">FLARE-001</div>
                  <div className="text-xs text-[#5f6368]">Stellar Flares</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">MICROLENS-001</div>
                  <div className="text-xs text-[#5f6368]">Microlensing</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">SUPERNOVA-001</div>
                  <div className="text-xs text-[#5f6368]">Transients</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">SPECTYPE-001</div>
                  <div className="text-xs text-[#5f6368]">Spectral Class</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">ASTERO-001</div>
                  <div className="text-xs text-[#5f6368]">Asteroseismology</div>
                </div>
                <div className="text-center p-3 bg-[#f8f9fa] rounded">
                  <div className="font-medium text-[#202124] mb-1">GALAXY-001</div>
                  <div className="text-xs text-[#5f6368]">Morphology</div>
                </div>
              </div>
            </div>
          </section>

          {/* First Steps Section */}
          <section id="first-steps" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">First Steps</h2>
            <p className="text-[#3c4043] mb-6">
              Getting started with LARUN.SPACE is straightforward. Follow these steps to run your first analysis on the Cloud platform.
            </p>

            <div className="space-y-4">
              {/* Step 1 */}
              <div className="bg-white border border-[#dadce0] rounded-lg p-5 flex gap-4">
                <div className="w-8 h-8 bg-[#202124] text-white rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0">
                  1
                </div>
                <div>
                  <h4 className="font-medium text-[#202124] mb-2">Sign Up Free</h4>
                  <p className="text-sm text-[#5f6368] mb-3">
                    Create a free account to get 5 analyses per month. No credit card required.
                  </p>
                  <Link href="/cloud/auth/signup" className="text-sm text-[#1a73e8] hover:underline">
                    Sign up now →
                  </Link>
                </div>
              </div>

              {/* Step 2 */}
              <div className="bg-white border border-[#dadce0] rounded-lg p-5 flex gap-4">
                <div className="w-8 h-8 bg-[#202124] text-white rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0">
                  2
                </div>
                <div>
                  <h4 className="font-medium text-[#202124] mb-2">Upload FITS File</h4>
                  <p className="text-sm text-[#5f6368] mb-3">
                    Upload your astronomical light curve data in FITS format. Our platform accepts standard FITS files
                    from TESS, Kepler, and other missions.
                  </p>
                </div>
              </div>

              {/* Step 3 */}
              <div className="bg-white border border-[#dadce0] rounded-lg p-5 flex gap-4">
                <div className="w-8 h-8 bg-[#202124] text-white rounded-full flex items-center justify-center text-sm font-medium flex-shrink-0">
                  3
                </div>
                <div>
                  <h4 className="font-medium text-[#202124] mb-2">Select Model & Analyze</h4>
                  <p className="text-sm text-[#5f6368] mb-3">
                    Choose from 8 specialized TinyML models. Each model is optimized for specific astronomical phenomena.
                    Results typically return in under 10 seconds with inference times &lt;10ms.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Understanding Results Section */}
          <section id="understanding-results" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">Understanding Results</h2>
            <p className="text-[#3c4043] mb-6">
              Learn how to interpret the output from Larun&apos;s detection system.
            </p>

            <div className="bg-white border border-[#dadce0] rounded-lg overflow-hidden mb-6">
              <table className="w-full text-sm">
                <thead className="bg-[#f1f3f4]">
                  <tr>
                    <th className="text-left p-4 font-medium text-[#202124]">Parameter</th>
                    <th className="text-left p-4 font-medium text-[#202124]">Description</th>
                    <th className="text-left p-4 font-medium text-[#202124]">Typical Range</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-[#dadce0]">
                    <td className="p-4 font-medium text-[#202124]">Confidence</td>
                    <td className="p-4 text-[#5f6368]">Model&apos;s certainty that a transit is present</td>
                    <td className="p-4 text-[#5f6368]">0-100%</td>
                  </tr>
                  <tr className="border-t border-[#dadce0]">
                    <td className="p-4 font-medium text-[#202124]">Period</td>
                    <td className="p-4 text-[#5f6368]">Estimated orbital period from BLS analysis</td>
                    <td className="p-4 text-[#5f6368]">0.5-50 days</td>
                  </tr>
                  <tr className="border-t border-[#dadce0]">
                    <td className="p-4 font-medium text-[#202124]">Depth</td>
                    <td className="p-4 text-[#5f6368]">Transit depth indicating planet size</td>
                    <td className="p-4 text-[#5f6368]">0.01-3%</td>
                  </tr>
                  <tr className="border-t border-[#dadce0]">
                    <td className="p-4 font-medium text-[#202124]">Duration</td>
                    <td className="p-4 text-[#5f6368]">Time the planet takes to cross the star</td>
                    <td className="p-4 text-[#5f6368]">1-12 hours</td>
                  </tr>
                  <tr className="border-t border-[#dadce0]">
                    <td className="p-4 font-medium text-[#202124]">SNR</td>
                    <td className="p-4 text-[#5f6368]">Signal-to-noise ratio of the detection</td>
                    <td className="p-4 text-[#5f6368]">&gt;7 for reliable</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>

          {/* Transit Search Section */}
          <section id="transit-search" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">Transit Search Workflow</h2>
            <p className="text-[#3c4043] mb-6">
              The transit search workflow uses the Box Least Squares (BLS) algorithm combined with machine learning
              to identify potential planetary transits in photometric data.
            </p>

            <div className="bg-white border border-[#dadce0] rounded-lg p-6 mb-6">
              <h3 className="text-lg font-medium text-[#202124] mb-4">How It Works</h3>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-[#e8f0fe] text-[#1a73e8] rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                    1
                  </div>
                  <div>
                    <span className="font-medium text-[#202124]">Data Retrieval:</span>
                    <span className="text-[#5f6368]"> Light curve fetched from MAST archive</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-[#e8f0fe] text-[#1a73e8] rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                    2
                  </div>
                  <div>
                    <span className="font-medium text-[#202124]">Preprocessing:</span>
                    <span className="text-[#5f6368]"> Outlier removal, detrending, normalization</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-[#e8f0fe] text-[#1a73e8] rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                    3
                  </div>
                  <div>
                    <span className="font-medium text-[#202124]">BLS Analysis:</span>
                    <span className="text-[#5f6368]"> Search for periodic box-shaped dips</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-[#e8f0fe] text-[#1a73e8] rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                    4
                  </div>
                  <div>
                    <span className="font-medium text-[#202124]">ML Classification:</span>
                    <span className="text-[#5f6368]"> TinyML model evaluates candidate signals</span>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-[#e8f0fe] text-[#1a73e8] rounded-full flex items-center justify-center text-xs font-medium flex-shrink-0 mt-0.5">
                    5
                  </div>
                  <div>
                    <span className="font-medium text-[#202124]">Results:</span>
                    <span className="text-[#5f6368]"> Confidence score and transit parameters</span>
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* Candidate Vetting Section */}
          <section id="candidate-vetting" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">Candidate Vetting</h2>
            <p className="text-[#3c4043] mb-6">
              Not all detections are real planets. The vetting process helps distinguish genuine planetary signals
              from false positives like eclipsing binaries, stellar variability, or instrumental artifacts.
            </p>

            <h3 className="text-lg font-medium text-[#202124] mb-4">Disposition Categories</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-white border border-[#1e8e3e] rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 bg-[#1e8e3e] rounded-full"></div>
                  <span className="font-medium text-[#1e8e3e]">PC (Planet Candidate)</span>
                </div>
                <p className="text-sm text-[#5f6368]">
                  Passes all vetting tests. Ready for follow-up observations and potential confirmation.
                </p>
              </div>
              <div className="bg-white border border-[#f9ab00] rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 bg-[#f9ab00] rounded-full"></div>
                  <span className="font-medium text-[#f9ab00]">APC (Ambiguous)</span>
                </div>
                <p className="text-sm text-[#5f6368]">
                  Some tests inconclusive. Requires additional analysis or observations.
                </p>
              </div>
              <div className="bg-white border border-[#d93025] rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-3 h-3 bg-[#d93025] rounded-full"></div>
                  <span className="font-medium text-[#d93025]">FP (False Positive)</span>
                </div>
                <p className="text-sm text-[#5f6368]">
                  Failed one or more vetting tests. Likely not a planetary transit.
                </p>
              </div>
            </div>

            <div className="bg-white border border-[#dadce0] rounded-lg p-6">
              <h3 className="text-lg font-medium text-[#202124] mb-4">Vetting Tests</h3>
              <ul className="space-y-3 text-sm">
                <li className="flex items-start gap-2">
                  <span className="text-[#1a73e8]">•</span>
                  <span><strong>Odd/Even Test:</strong> Compares transit depths of odd and even transits to detect eclipsing binaries</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#1a73e8]">•</span>
                  <span><strong>Secondary Eclipse:</strong> Searches for a secondary dip indicating a self-luminous companion</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#1a73e8]">•</span>
                  <span><strong>Centroid Motion:</strong> Checks if the signal comes from the target star or a nearby source</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#1a73e8]">•</span>
                  <span><strong>Transit Shape:</strong> Verifies the transit has the expected limb-darkening profile</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#1a73e8]">•</span>
                  <span><strong>Stellar Parameters:</strong> Confirms derived planet size is physically plausible</span>
                </li>
              </ul>
            </div>
          </section>

          {/* Habitability Section */}
          <section id="habitability" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">Habitability Analysis</h2>
            <p className="text-[#3c4043] mb-6">
              For validated candidates, Larun can estimate habitability potential based on orbital parameters
              and stellar characteristics.
            </p>

            <div className="bg-white border border-[#dadce0] rounded-lg p-6 mb-6">
              <h3 className="text-lg font-medium text-[#202124] mb-4">Key Factors</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 bg-[#f1f3f4] rounded-lg">
                  <h4 className="font-medium text-[#202124] mb-2">Habitable Zone</h4>
                  <p className="text-sm text-[#5f6368]">
                    Is the planet in the region where liquid water could exist on the surface?
                  </p>
                </div>
                <div className="p-4 bg-[#f1f3f4] rounded-lg">
                  <h4 className="font-medium text-[#202124] mb-2">Planet Size</h4>
                  <p className="text-sm text-[#5f6368]">
                    Rocky planets (0.5-1.5 Earth radii) are more likely to be habitable.
                  </p>
                </div>
                <div className="p-4 bg-[#f1f3f4] rounded-lg">
                  <h4 className="font-medium text-[#202124] mb-2">Stellar Type</h4>
                  <p className="text-sm text-[#5f6368]">
                    K and G-type stars provide stable conditions for long periods.
                  </p>
                </div>
                <div className="p-4 bg-[#f1f3f4] rounded-lg">
                  <h4 className="font-medium text-[#202124] mb-2">Orbital Eccentricity</h4>
                  <p className="text-sm text-[#5f6368]">
                    Circular orbits provide more stable temperature conditions.
                  </p>
                </div>
              </div>
            </div>
          </section>

          {/* Glossary Section */}
          <section id="glossary" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">Glossary</h2>
            <div className="bg-white border border-[#dadce0] rounded-lg divide-y divide-[#dadce0]">
              <div className="p-4">
                <dt className="font-medium text-[#202124]">BLS (Box Least Squares)</dt>
                <dd className="text-sm text-[#5f6368] mt-1">Algorithm for detecting periodic box-shaped dips in time series data</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">Light Curve</dt>
                <dd className="text-sm text-[#5f6368] mt-1">A graph of brightness vs. time showing how a star&apos;s light varies</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">Transit</dt>
                <dd className="text-sm text-[#5f6368] mt-1">When a planet passes in front of its star, causing a temporary dimming</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">TIC</dt>
                <dd className="text-sm text-[#5f6368] mt-1">TESS Input Catalog - database of stars observed by TESS</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">KIC</dt>
                <dd className="text-sm text-[#5f6368] mt-1">Kepler Input Catalog - database of stars observed by Kepler</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">SNR</dt>
                <dd className="text-sm text-[#5f6368] mt-1">Signal-to-Noise Ratio - measure of signal strength relative to noise</dd>
              </div>
              <div className="p-4">
                <dt className="font-medium text-[#202124]">MAST</dt>
                <dd className="text-sm text-[#5f6368] mt-1">Mikulski Archive for Space Telescopes - NASA&apos;s data archive</dd>
              </div>
            </div>
          </section>

          {/* References Section */}
          <section id="references" className="mb-12">
            <h2 className="text-2xl font-normal text-[#202124] mb-4 pt-6 border-t border-[#dadce0]">References</h2>
            <div className="bg-white border border-[#dadce0] rounded-lg p-6">
              <ul className="space-y-3 text-sm">
                <li>
                  <a href="https://exoplanets.nasa.gov/" target="_blank" rel="noopener noreferrer" className="text-[#1a73e8] no-underline hover:underline">
                    NASA Exoplanet Exploration
                  </a>
                  <span className="text-[#5f6368]"> - Official NASA resource for exoplanet science</span>
                </li>
                <li>
                  <a href="https://tess.mit.edu/" target="_blank" rel="noopener noreferrer" className="text-[#1a73e8] no-underline hover:underline">
                    TESS Mission
                  </a>
                  <span className="text-[#5f6368]"> - Transiting Exoplanet Survey Satellite</span>
                </li>
                <li>
                  <a href="https://archive.stsci.edu/" target="_blank" rel="noopener noreferrer" className="text-[#1a73e8] no-underline hover:underline">
                    MAST Archive
                  </a>
                  <span className="text-[#5f6368]"> - Space telescope data repository</span>
                </li>
                <li>
                  <a href="https://exofop.ipac.caltech.edu/tess/" target="_blank" rel="noopener noreferrer" className="text-[#1a73e8] no-underline hover:underline">
                    ExoFOP-TESS
                  </a>
                  <span className="text-[#5f6368]"> - Follow-up observation coordination</span>
                </li>
              </ul>
            </div>
          </section>

          {/* CTA Section */}
          <div className="bg-gradient-to-r from-[#1a73e8] to-[#174ea6] rounded-lg p-8 text-center">
            <h3 className="text-xl font-medium text-white mb-2">Ready to Start?</h3>
            <p className="text-white opacity-90 mb-6">
              Sign up free and start analyzing astronomical data with TinyML models.
            </p>
            <Link
              href="/cloud/auth/signup"
              className="inline-block px-6 py-3 bg-white text-[#1a73e8] text-sm font-medium rounded no-underline hover:bg-gray-100 transition-colors"
            >
              Start Free Trial
            </Link>
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="ml-[260px] py-8 border-t border-[#dadce0] bg-white text-center">
        <div className="flex justify-center gap-6 mb-4">
          <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            laruneng.com
          </a>
          <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            GitHub
          </a>
          <Link href="/faq" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            FAQ
          </Link>
          <Link href="/cloud/pricing" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            Pricing
          </Link>
        </div>
        <p className="text-xs text-[#5f6368]">&copy; {new Date().getFullYear()} Larun.Space. All rights reserved.</p>
      </footer>
    </div>
  );
}
