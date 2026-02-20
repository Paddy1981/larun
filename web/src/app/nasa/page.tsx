'use client';

import { useState } from 'react';
import Link from 'next/link';
import Header from '@/components/Header';

// Helper: generate NASA document URL
function nasaStdUrl(code: string): string {
  const c = code.toLowerCase().replace(/[.\s]/g, '-').replace(/[^a-z0-9-]/g, '');
  if (code.startsWith('NPR') || code.startsWith('NPD')) {
    return 'https://nodis3.gsfc.nasa.gov/';
  }
  return `https://standards.nasa.gov/standard/nasa/${c}`;
}

function NasaDocLink({ code, children }: { code: string; children?: React.ReactNode }) {
  return (
    <a
      href={nasaStdUrl(code)}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-[13px] bg-[#e8eeff] text-[#0b3d91] hover:bg-[#0b3d91] hover:text-white px-2 py-0.5 rounded transition-colors cursor-pointer"
      title={`View ${code}`}
    >
      {children || code}
      <svg className="w-3 h-3 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </a>
  );
}

function NprLink({ code, children }: { code: string; children?: React.ReactNode }) {
  return (
    <a
      href="https://nodis3.gsfc.nasa.gov/"
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-[13px] bg-[#fef3c7] text-[#92400e] hover:bg-[#92400e] hover:text-white px-2 py-0.5 rounded transition-colors cursor-pointer"
      title={`View ${code} on NODIS`}
    >
      {children || code}
      <svg className="w-3 h-3 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </a>
  );
}

const NAV_SECTIONS = [
  { id: 'overview',       label: 'Overview',              icon: '◎' },
  { id: 'governance',     label: 'Governance Framework',  icon: '⚖' },
  { id: 'lifecycle',      label: 'Lifecycle & Reviews',   icon: '↺' },
  { id: 'structures',     label: '5000 — Structures',     icon: '▦' },
  { id: 'electrical',     label: '4000 — Electrical',     icon: '⚡' },
  { id: 'materials',      label: '6000 — Materials',      icon: '◈' },
  { id: 'test-env',       label: '7000 — Test/Env',       icon: '🧪' },
  { id: 'safety',         label: '8000 — Safety',         icon: '⚠' },
  { id: 'software',       label: 'Software',              icon: '⌨' },
  { id: 'human',          label: 'Human Systems',         icon: '👤' },
  { id: 'commercial',     label: 'Commercial Programs',   icon: '🚀' },
  { id: 'artemis',        label: 'Artemis & Exploration', icon: '🌙' },
  { id: 'complete-list',  label: 'Complete List',         icon: '≡' },
  { id: 'npr-index',      label: 'NPR / NPD Index',       icon: 'P' },
  { id: 'crossref',       label: 'Cross-Reference',       icon: '⇔' },
  { id: 'glossary',       label: 'Glossary',              icon: 'G' },
  { id: 'resources',      label: 'Resources & Centers',   icon: '↗' },
];

export default function NasaPage() {
  const [activeSection, setActiveSection] = useState('overview');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const scrollTo = (id: string) => {
    setActiveSection(id);
    setSidebarOpen(false);
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  return (
    <div className="min-h-screen bg-white text-[#202124]">
      <Header />

      {/* Mobile toggle */}
      <button
        className="fixed bottom-6 right-6 z-50 md:hidden bg-[#0b3d91] text-white w-12 h-12 rounded-full shadow-lg flex items-center justify-center"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className={`fixed top-16 left-0 bottom-0 w-64 bg-[#f8f9fa] border-r border-[#dadce0] z-40 overflow-y-auto transition-transform duration-300 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          {/* Header */}
          <div className="p-5 border-b border-[#dadce0]">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-[#0b3d91] rounded-lg flex items-center justify-center text-white font-bold text-sm">N</div>
              <div>
                <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider">NASA</p>
                <p className="text-sm font-bold text-[#202124]">Standards Hub</p>
              </div>
            </div>
          </div>

          {/* Stats */}
          <div className="flex divide-x divide-[#dadce0] border-b border-[#dadce0]">
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">83</p>
              <p className="text-[10px] text-[#5f6368]">Documents</p>
            </div>
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">10</p>
              <p className="text-[10px] text-[#5f6368]">Centers</p>
            </div>
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">7</p>
              <p className="text-[10px] text-[#5f6368]">Disciplines</p>
            </div>
          </div>

          {/* Nav */}
          <nav className="p-3">
            {NAV_SECTIONS.map(s => (
              <button
                key={s.id}
                onClick={() => scrollTo(s.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-left transition-colors mb-0.5 ${
                  activeSection === s.id
                    ? 'bg-[#e8eeff] text-[#0b3d91] font-semibold'
                    : 'text-[#3c4043] hover:bg-[#f1f3f4]'
                }`}
              >
                <span className="w-5 h-5 rounded text-center leading-5 text-xs font-mono flex-shrink-0">{s.icon}</span>
                {s.label}
              </button>
            ))}
          </nav>

          <div className="p-4 border-t border-[#dadce0] mt-2">
            <a
              href="https://standards.nasa.gov"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-[#0b3d91] hover:underline"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
              </svg>
              standards.nasa.gov ↗
            </a>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 md:ml-64 min-h-screen">
          {/* Hero */}
          <div className="bg-gradient-to-r from-[#0b3d91] to-[#1a237e] text-white px-8 py-12">
            <div className="max-w-4xl">
              <div className="inline-flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full text-xs font-medium mb-4">
                NASA Technical Standards Program (NTSP)
              </div>
              <h1 className="text-3xl md:text-4xl font-bold mb-3">NASA Standards Hub</h1>
              <p className="text-white/85 text-lg max-w-2xl">
                The complete reference for 83 NASA-STD, NASA-HDBK, and NASA-SPEC documents across 7 engineering discipline categories — from structures to human spaceflight.
              </p>
              <div className="flex flex-wrap gap-3 mt-6">
                {['NASA-STD (≈45)', 'NASA-HDBK (≈37)', 'NASA-SPEC (1)', 'NPR/NPD directives'].map(t => (
                  <span key={t} className="bg-white/15 text-white text-xs px-3 py-1.5 rounded-full font-medium">{t}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="px-6 md:px-10 py-10 max-w-4xl">

            {/* ========== OVERVIEW ========== */}
            <section id="overview" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Overview</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <p className="text-[#3c4043] mb-6">
                The NASA Technical Standards Program (NTSP) maintains the Agency's collection of technical standards, handbooks, and specifications. These documents establish mandatory requirements (NASA-STD) and guidance (NASA-HDBK) for all NASA programs and projects, complemented by NPRs (Procedural Requirements) and NPDs (Policy Directives) in NODIS.
              </p>

              <div className="grid md:grid-cols-3 gap-4 mb-8">
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Document Types</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs font-mono bg-[#e8eeff] text-[#0b3d91] px-1.5 py-0.5 rounded text-center">NASA-STD</span>
                      <span className="text-[#3c4043]">Mandatory "shall" requirements</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs font-mono bg-[#fef3c7] text-[#92400e] px-1.5 py-0.5 rounded text-center">NASA-HDBK</span>
                      <span className="text-[#3c4043]">Guidance "should" recommendations</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs font-mono bg-[#d1fae5] text-[#065f46] px-1.5 py-0.5 rounded text-center">NPR/NPD</span>
                      <span className="text-[#3c4043]">Procedural requirements &amp; policy</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-20 text-xs font-mono bg-[#fee2e2] text-[#991b1b] px-1.5 py-0.5 rounded text-center">NASA-SPEC</span>
                      <span className="text-[#3c4043]">Product specifications</span>
                    </div>
                  </div>
                </div>

                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Numbering System</h3>
                  <div className="space-y-2 text-xs font-mono text-[#3c4043]">
                    <p><span className="text-[#0b3d91] font-bold">1000–1999</span> Program Mgmt</p>
                    <p><span className="text-[#0b3d91] font-bold">2000–2999</span> Science</p>
                    <p><span className="text-[#0b3d91] font-bold">3000–3999</span> Human Systems</p>
                    <p><span className="text-[#0b3d91] font-bold">4000–4999</span> Electrical</p>
                    <p><span className="text-[#0b3d91] font-bold">5000–5999</span> Structures/Mech</p>
                    <p><span className="text-[#0b3d91] font-bold">6000–6999</span> Materials</p>
                    <p><span className="text-[#0b3d91] font-bold">7000–7999</span> Test &amp; Environment</p>
                    <p><span className="text-[#0b3d91] font-bold">8000–8999</span> Safety</p>
                  </div>
                </div>

                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Access</h3>
                  <ul className="text-sm text-[#3c4043] space-y-2">
                    <li className="flex gap-2"><span className="text-[#0b3d91]">✓</span>"Internet Public" — freely downloadable</li>
                    <li className="flex gap-2"><span className="text-[#5f6368]">⊘</span>Some require NASA internal access</li>
                    <li className="flex gap-2"><span className="text-[#0b3d91]">✓</span>NTSS provides update notifications</li>
                    <li className="flex gap-2"><span className="text-[#0b3d91]">✓</span>Feedback via standards.nasa.gov</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* ========== GOVERNANCE ========== */}
            <section id="governance" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Governance Framework</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">NASA governance flows from the Office of the Chief Engineer (OCE), Office of Safety &amp; Mission Assurance (OSMA), and the Technical Authority (TA) system. Standards are mandatory unless formally waived.</p>

              <div className="grid md:grid-cols-2 gap-5 mb-6">
                {[
                  { code: 'NPR 7120.5F', title: 'Space Flight Program & Project Management', desc: 'Top-level governance for all space flight programs. Defines lifecycle, KDPs, reviews, and roles.' },
                  { code: 'NPR 7123.1C', title: 'Systems Engineering Processes & Requirements', desc: 'NASA\'s systems engineering framework — 17 processes covering technical requirements, design, integration, and verification.' },
                  { code: 'NPR 7150.2C', title: 'Software Engineering Requirements', desc: 'Mandatory software engineering practices, classification levels A–E, and documentation requirements.' },
                  { code: 'NPR 8000.4', title: 'Agency Risk Management Procedural Requirements', desc: 'Risk management framework — identification, analysis, handling, and monitoring across all programs.' },
                ].map(item => (
                  <div key={item.code} className="border border-[#dadce0] rounded-xl p-5 hover:border-[#0b3d91] transition-colors">
                    <div className="mb-2"><NprLink code={item.code} /></div>
                    <h3 className="font-semibold text-[#202124] text-sm mb-1">{item.title}</h3>
                    <p className="text-xs text-[#5f6368]">{item.desc}</p>
                  </div>
                ))}
              </div>

              <div className="bg-[#e8eeff] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e3a8a]">
                <strong>Technical Authority:</strong> NASA uses a dual-authority model — Program/Project Management Authority (PMA) for cost/schedule, and Independent Technical Authority (ITA) for technical decisions. ITA can halt work and cannot be overridden by program managers on safety issues.
              </div>
            </section>

            {/* ========== LIFECYCLE ========== */}
            <section id="lifecycle" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Lifecycle &amp; Reviews</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">NASA projects follow a phased lifecycle with Key Decision Points (KDPs) and Life-Cycle Reviews (LCRs) at each phase boundary, governed by <NprLink code="NPR 7120.5F" />.</p>

              <div className="space-y-3 mb-8">
                {[
                  { phase: 'Pre-A', name: 'Concept Studies', reviews: 'MCR', color: '#dbeafe' },
                  { phase: 'A', name: 'Concept & Technology Development', reviews: 'SRR', color: '#dbeafe' },
                  { phase: 'B', name: 'Preliminary Design & Technology Completion', reviews: 'MDR/SDR, PDR', color: '#e0e7ff' },
                  { phase: 'C', name: 'Final Design & Fabrication', reviews: 'CDR, SIR', color: '#ede9fe' },
                  { phase: 'D', name: 'System Assembly, Integration & Test, Launch', reviews: 'TRR, SAR, FRR', color: '#fef3c7' },
                  { phase: 'E', name: 'Operations & Sustainment', reviews: 'ORR, DR', color: '#d1fae5' },
                  { phase: 'F', name: 'Closeout', reviews: 'EOM', color: '#fee2e2' },
                ].map(p => (
                  <div key={p.phase} className="flex items-center gap-4 p-4 border border-[#dadce0] rounded-lg">
                    <span className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0 text-[#202124]" style={{ background: p.color }}>{p.phase}</span>
                    <div className="flex-1">
                      <p className="font-medium text-[#202124] text-sm">{p.name}</p>
                    </div>
                    <span className="text-xs font-mono text-[#5f6368] bg-[#f8f9fa] border border-[#dadce0] px-2 py-1 rounded">{p.reviews}</span>
                  </div>
                ))}
              </div>
            </section>

            {/* ========== STRUCTURES 5000 ========== */}
            <section id="structures" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">5000 Series — Structures &amp; Mechanical</h2>
              <div className="h-1 w-12 bg-[#3b82f6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Structural design, fracture control, pressure vessels, fatigue, and mechanisms standards for spaceflight hardware.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-24">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-STD-5001B', title: 'Structural Design and Test Factors for Spaceflight Hardware', type: 'STD' },
                      { code: 'NASA-STD-5002', title: 'Load Analyses of Spacecraft and Payloads', type: 'STD' },
                      { code: 'NASA-STD-5003', title: 'Fracture Control Requirements for Payloads Using the Space Shuttle', type: 'STD' },
                      { code: 'NASA-STD-5017', title: 'Design and Development Requirements for Mechanisms', type: 'STD' },
                      { code: 'NASA-STD-5019A', title: 'Fracture Control Requirements for Spaceflight Hardware', type: 'STD' },
                      { code: 'NASA-STD-5020A', title: 'Requirements for Threaded Fastening Systems in Spaceflight Hardware', type: 'STD' },
                      { code: 'NASA-HDBK-5010', title: 'Fracture Control Implementation Handbook', type: 'HDBK' },
                      { code: 'NASA-HDBK-5012', title: 'Handbook for Spacecraft Structural Dynamics Testing', type: 'HDBK' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><NasaDocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${row.type === 'STD' ? 'bg-[#dbeafe] text-[#1e40af]' : 'bg-[#fef3c7] text-[#92400e]'}`}>{row.type}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== ELECTRICAL 4000 ========== */}
            <section id="electrical" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">4000 Series — Electrical</h2>
              <div className="h-1 w-12 bg-[#f59e0b] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Electrical bonding, wiring, EEE parts, and printed wiring board standards for space hardware.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-24">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-STD-4003A', title: 'Electrical Bonding for NASA Launch Vehicles, Spacecraft, Payloads, and Flight Equipment', type: 'STD' },
                      { code: 'NASA-STD-4005B', title: 'Low Earth Orbit Spacecraft Charging Design Standard', type: 'STD' },
                      { code: 'NASA-HDBK-4001A', title: 'Electrical Grounding Architecture for Unmanned Spacecraft', type: 'HDBK' },
                      { code: 'NASA-HDBK-4002B', title: 'Mitigating In-Space Charging Effects', type: 'HDBK' },
                      { code: 'NASA-STD-8739.4A', title: 'Crimping, Interconnecting Cables, Harnesses and Wiring', type: 'STD' },
                      { code: 'NASA-STD-8739.5', title: 'Fiber Optic Terminations, Cable Assemblies and Installation', type: 'STD' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><NasaDocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${row.type === 'STD' ? 'bg-[#dbeafe] text-[#1e40af]' : 'bg-[#fef3c7] text-[#92400e]'}`}>{row.type}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== MATERIALS 6000 ========== */}
            <section id="materials" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">6000 Series — Materials &amp; Processes</h2>
              <div className="h-1 w-12 bg-[#10b981] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Materials selection, flammability, toxicity, offgassing, coatings, welding, and cleanliness requirements for spaceflight hardware.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-24">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-STD-6001C', title: 'Flammability, Offgassing, and Compatibility Requirements and Test Procedures', type: 'STD' },
                      { code: 'NASA-STD-6002C', title: 'Applying Data Matrix Identification Symbols', type: 'STD' },
                      { code: 'NASA-STD-6008A', title: 'NASA Fastener Procurement, Inspection, and Storage Requirements', type: 'STD' },
                      { code: 'NASA-STD-6016C', title: 'Standard Materials and Processes Requirements for Spacecraft', type: 'STD' },
                      { code: 'NASA-HDBK-6011', title: 'Aerospace Threaded Fastener Strength in Combined Shear & Tension Loading', type: 'HDBK' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><NasaDocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${row.type === 'STD' ? 'bg-[#dbeafe] text-[#1e40af]' : 'bg-[#fef3c7] text-[#92400e]'}`}>{row.type}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== TEST & ENVIRONMENT 7000 ========== */}
            <section id="test-env" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">7000 Series — Test &amp; Environments</h2>
              <div className="h-1 w-12 bg-[#8b5cf6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Environmental test requirements (vibration, shock, thermal vacuum, acoustic) and qualification/acceptance testing standards.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-24">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-HDBK-7005', title: 'Dynamic Environmental Criteria', type: 'HDBK' },
                      { code: 'NASA-HDBK-7008', title: 'Spacecraft Level Dynamic Environments Testing', type: 'HDBK' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><NasaDocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-[#fef3c7] text-[#92400e]">{row.type}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== SAFETY 8000 ========== */}
            <section id="safety" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">8000 Series — Safety &amp; Mission Assurance</h2>
              <div className="h-1 w-12 bg-[#ef4444] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Safety, reliability, quality assurance, orbital debris, workmanship, and probabilistic risk assessment standards.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-24">Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-STD-8719.13C', title: 'Software Safety Standard', type: 'STD' },
                      { code: 'NASA-STD-8719.17D', title: 'NASA Requirements for Limiting Orbital Debris', type: 'STD' },
                      { code: 'NASA-STD-8719.24A', title: 'NASA Expendable Launch Vehicle Payload Safety Requirements', type: 'STD' },
                      { code: 'NASA-STD-8729.1B', title: 'Planning, Developing, and Managing an Effective Reliability & Maintainability Program', type: 'STD' },
                      { code: 'NASA-STD-8739.8B', title: 'Software Assurance Standard', type: 'STD' },
                      { code: 'NASA-HDBK-8739.19', title: 'NASA Measurement System Validation Practitioner\'s Handbook', type: 'HDBK' },
                      { code: 'NPR 8715.3C', title: 'NASA General Safety Program Requirements', type: 'NPR' },
                      { code: 'NPR 8705.5', title: 'Technical Probabilistic Risk Assessment (PRA) Procedures', type: 'NPR' },
                      { code: 'NPR 8705.6', title: 'Safety and Mission Assurance Audits, Reviews, and Assessments', type: 'NPR' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          {row.type === 'NPR' ? <NprLink code={row.code} /> : <NasaDocLink code={row.code} />}
                        </td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                            row.type === 'STD' ? 'bg-[#dbeafe] text-[#1e40af]' :
                            row.type === 'HDBK' ? 'bg-[#fef3c7] text-[#92400e]' :
                            'bg-[#d1fae5] text-[#065f46]'
                          }`}>{row.type}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== SOFTWARE ========== */}
            <section id="software" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Software Standards</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">NASA software standards span engineering (NPR 7150.2C), assurance (NASA-STD-8739.8B), and safety (NASA-STD-8719.13C), with classification levels A–E by criticality.</p>

              <div className="grid md:grid-cols-3 gap-4 mb-8">
                {[
                  { code: 'NPR 7150.2C', title: 'Software Engineering Requirements', desc: 'Mandatory software engineering processes, documentation, and independent verification requirements.' },
                  { code: 'NASA-STD-8739.8B', title: 'Software Assurance Standard', desc: 'Software product assurance planning, reviews, audits, and supplier qualification requirements.' },
                  { code: 'NASA-STD-8719.13C', title: 'Software Safety Standard', desc: 'Safety-critical software identification, analysis, and design requirements including hazard analysis.' },
                ].map(item => (
                  <div key={item.code} className="border border-[#dadce0] rounded-xl p-5">
                    <div className="mb-2">
                      {item.code.startsWith('NPR') ? <NprLink code={item.code} /> : <NasaDocLink code={item.code} />}
                    </div>
                    <h3 className="font-semibold text-[#202124] text-sm mb-2">{item.title}</h3>
                    <p className="text-xs text-[#5f6368]">{item.desc}</p>
                  </div>
                ))}
              </div>

              <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                <h3 className="font-semibold text-[#202124] mb-3">Software Classification (NPR 7150.2C)</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="bg-white text-left">
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Class</th>
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Criticality</th>
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Example</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { cls: 'A', crit: 'Catastrophic failure impact', ex: 'Human-rated flight control' },
                        { cls: 'B', crit: 'Critical failure impact', ex: 'Mission critical systems' },
                        { cls: 'C', crit: 'Significant failure impact', ex: 'Science payload software' },
                        { cls: 'D', crit: 'Negligible failure impact', ex: 'Ground tools' },
                        { cls: 'E', crit: 'Information only (legacy/COTS)', ex: 'Off-the-shelf OS' },
                      ].map((row, i) => (
                        <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] font-bold text-[#0b3d91]">{row.cls}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#3c4043]">{row.crit}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#5f6368]">{row.ex}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* ========== HUMAN SYSTEMS ========== */}
            <section id="human" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Human Systems Integration</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Standards for crewed spaceflight — habitability, human factors, EVA, medical, and human-rating requirements.</p>

              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NASA-STD-3001 Vol.1', title: 'NASA Space Flight Human-System Standard: Crew Health' },
                      { code: 'NASA-STD-3001 Vol.2', title: 'NASA Space Flight Human-System Standard: Human Factors, Habitability, and Environmental Health' },
                      { code: 'NASA-HDBK-3000', title: 'NASA Man-Systems Integration Standards (MSIS)' },
                      { code: 'NPR 8705.2C', title: 'Human-Rating Requirements for Space Systems' },
                      { code: 'NPR 8900.1', title: 'Health and Medical Requirements for Human Space Exploration' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          {row.code.startsWith('NPR') ? <NprLink code={row.code} /> : <NasaDocLink code={row.code} />}
                        </td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-[#e8eeff] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e3a8a]">
                <strong>Human Rating:</strong> <NprLink code="NPR 8705.2C" /> requires a Human Rating Certification Package (HRCP) covering design, analysis, test, and operations evidence. Four certification themes: Safety, Reliability, Operability, and Human Factors.
              </div>
            </section>

            {/* ========== COMMERCIAL ========== */}
            <section id="commercial" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Commercial Programs</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">NASA's commercial partnership framework uses Space Act Agreements (SAAs) and CCtCap-style performance-based requirements rather than full FAR/DFARS compliance.</p>

              <div className="grid md:grid-cols-2 gap-5 mb-6">
                {[
                  { name: 'Commercial Crew Program (CCP)', badge: 'CCP', desc: 'Dragon (SpaceX) and Starliner (Boeing) certified for ISS crew rotation. CTS certification: Programmatic, Design, Production, Operations elements.', color: '#dbeafe', textColor: '#1e40af' },
                  { name: 'Commercial Resupply Services (CRS)', badge: 'CRS', desc: 'CRS1 (Dragon, Cygnus) and CRS2 (Dragon, Cygnus, Dream Chaser) for ISS cargo. Performance-based contract requirements.', color: '#d1fae5', textColor: '#065f46' },
                  { name: 'Commercial LEO Destinations (CLD)', badge: 'CLD', desc: 'Next-gen private space stations to replace ISS. Axiom, Starlab, Orbital Reef selected. SAA-based development framework.', color: '#ede9fe', textColor: '#4c1d95' },
                  { name: 'Commercial Lunar Payload Services (CLPS)', badge: 'CLPS', desc: 'Small lunar landers (Astrobotic, Intuitive Machines, etc.) for science/tech delivery. Catalog-based task orders.', color: '#fef3c7', textColor: '#92400e' },
                ].map(p => (
                  <div key={p.name} className="border border-[#dadce0] rounded-xl p-5">
                    <div className="flex items-center gap-3 mb-3">
                      <span className="px-2.5 py-1 rounded-full text-xs font-bold" style={{ background: p.color, color: p.textColor }}>{p.badge}</span>
                      <h3 className="font-semibold text-[#202124] text-sm">{p.name}</h3>
                    </div>
                    <p className="text-sm text-[#5f6368]">{p.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* ========== ARTEMIS ========== */}
            <section id="artemis" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Artemis &amp; Exploration Programs</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">NASA's deep space exploration architecture — returning humans to the Moon and preparing for Mars.</p>

              <div className="grid md:grid-cols-2 gap-5">
                {[
                  { name: 'Space Launch System (SLS)', codes: ['NASA-STD-5001B', 'NASA-STD-5019A'], desc: 'Heavy-lift rocket — Block 1 (95 mT LEO), evolving to Block 2 (130 mT). Managed by Marshall Space Flight Center.', color: '#f97316' },
                  { name: 'Orion Multi-Purpose Crew Vehicle', codes: ['NASA-STD-3001 Vol.1', 'NPR 8705.2C'], desc: 'Deep space crew capsule with ESA-built Service Module. Human-rated per NPR 8705.2C. Managed by Johnson Space Center.', color: '#f59e0b' },
                  { name: 'Gateway', codes: [], desc: 'Lunar orbiting platform — HALO + PPE modules. International partnership (ESA, JAXA, CSA). Enables sustained lunar exploration.', color: '#8b5cf6' },
                  { name: 'Human Landing System (HLS)', codes: [], desc: 'Commercial lunar landers: SpaceX Starship HLS (initial), Blue Origin Blue Moon (sustaining).', color: '#10b981' },
                ].map(item => (
                  <div key={item.name} className="border border-[#dadce0] rounded-xl p-5" style={{ borderTopColor: item.color, borderTopWidth: 3 }}>
                    <h3 className="font-semibold text-[#202124] mb-2">{item.name}</h3>
                    <p className="text-sm text-[#5f6368] mb-3">{item.desc}</p>
                    {item.codes.length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {item.codes.map(c => (
                          c.startsWith('NPR') ? <NprLink key={c} code={c} /> : <NasaDocLink key={c} code={c} />
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

            {/* ========== COMPLETE LIST ========== */}
            <section id="complete-list" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Complete Standards List</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">
                All 83 active NASA Technical Standards, Handbooks, and Specifications are available at{' '}
                <a href="https://standards.nasa.gov/all-standards" target="_blank" rel="noopener noreferrer" className="text-[#0b3d91] hover:underline font-medium">standards.nasa.gov/all-standards ↗</a>.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Document Type</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Prefix</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Count</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Nature</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { type: 'Technical Standards', prefix: 'NASA-STD-', count: '≈45', nature: 'Mandatory requirements ("shall" statements)' },
                      { type: 'Technical Handbooks', prefix: 'NASA-HDBK-', count: '≈37', nature: 'Guidance ("should" recommendations)' },
                      { type: 'Specifications', prefix: 'NASA-SPEC-', count: '1', nature: 'Detailed product specification (Pyrovalves)' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.type}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-mono text-[#0b3d91] text-xs">{row.prefix}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-bold text-[#202124]">{row.count}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#5f6368] text-xs">{row.nature}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== NPR INDEX ========== */}
            <section id="npr-index" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Key NPR / NPD Index</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">The most frequently referenced NASA Procedural Requirements and Policy Directives, available at{' '}
                <a href="https://nodis3.gsfc.nasa.gov" target="_blank" rel="noopener noreferrer" className="text-[#0b3d91] hover:underline">nodis3.gsfc.nasa.gov ↗</a>.
              </p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-36">Document</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-32">Domain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'NPD 1000.0', title: 'NASA Governance and Strategic Management Handbook', domain: 'Governance' },
                      { code: 'NPD 7120.4', title: 'NASA Engineering and Program/Project Management Policy', domain: 'Policy' },
                      { code: 'NPR 7120.5F', title: 'NASA Space Flight Program and Project Management Requirements', domain: 'Program Mgmt' },
                      { code: 'NPR 7120.7', title: 'NASA IT and Institutional Infrastructure PM Requirements', domain: 'IT programs' },
                      { code: 'NPR 7120.8', title: 'NASA Research and Technology PM Requirements', domain: 'R&T programs' },
                      { code: 'NPR 7120.10B', title: 'Technical Standards for NASA Programs and Projects', domain: 'Standards gov.' },
                      { code: 'NPR 7123.1C', title: 'NASA Systems Engineering Processes and Requirements', domain: 'Sys. Engineering' },
                      { code: 'NPR 7150.2C', title: 'NASA Software Engineering Requirements', domain: 'Software' },
                      { code: 'NPR 8000.4', title: 'Agency Risk Management Procedural Requirements', domain: 'Risk Mgmt' },
                      { code: 'NPR 8705.2C', title: 'Human-Rating Requirements for Space Systems', domain: 'Human Rating' },
                      { code: 'NPR 8705.4', title: 'Risk Classification for NASA Payloads', domain: 'Risk class.' },
                      { code: 'NPR 8705.5', title: 'Technical PRA Procedures for Safety and Mission Success', domain: 'PRA' },
                      { code: 'NPR 8705.6', title: 'SMA Audits, Reviews, and Assessments', domain: 'SMA oversight' },
                      { code: 'NPR 8715.3C', title: 'NASA General Safety Program Requirements', domain: 'Safety' },
                      { code: 'NPR 8735.2', title: 'Management of QA Functions for NASA Contracts', domain: 'Quality' },
                      { code: 'NPR 8900.1', title: 'Health and Medical Requirements for Human Space Exploration', domain: 'Crew health' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><NprLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#5f6368] text-xs">{row.domain}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== CROSS-REFERENCE ========== */}
            <section id="crossref" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Cross-Reference: NASA ↔ ECSS ↔ Industry</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Domain</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">NASA</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">ECSS</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Industry</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { domain: 'Systems Engineering', nasa: 'NPR 7123.1C', ecss: 'ECSS-E-ST-10C Rev.1', ind: 'ISO/IEC 15288, INCOSE SE HB' },
                      { domain: 'Program Management', nasa: 'NPR 7120.5F', ecss: 'ECSS-M-ST-10C Rev.1', ind: 'PMI PMBOK' },
                      { domain: 'Human Rating', nasa: 'NPR 8705.2C', ecss: '—', ind: 'FAA 14 CFR Part 460' },
                      { domain: 'Software Engineering', nasa: 'NPR 7150.2C', ecss: 'ECSS-E-ST-40C Rev.1', ind: 'DO-178C, IEEE 12207' },
                      { domain: 'Software Assurance', nasa: 'NASA-STD-8739.8B', ecss: 'ECSS-Q-ST-80C Rev.1', ind: 'DO-178C, IEC 61508-3' },
                      { domain: 'Structural Design', nasa: 'NASA-STD-5001B', ecss: 'ECSS-E-ST-32C Rev.1', ind: 'AIAA S-110, S-111' },
                      { domain: 'Fracture Control', nasa: 'NASA-STD-5019A', ecss: 'ECSS-E-ST-32-01C Rev.2', ind: 'MIL-STD-1530' },
                      { domain: 'Safety', nasa: 'NPR 8715.3C', ecss: 'ECSS-Q-ST-40C Rev.1', ind: 'MIL-STD-882E, IEC 61508' },
                      { domain: 'FMEA/FMECA', nasa: 'NPR 8715.3 (ref)', ecss: 'ECSS-Q-ST-30-02C', ind: 'MIL-STD-1629, IEC 60812' },
                      { domain: 'Fault Tree', nasa: 'NASA Fault Tree HB', ecss: 'ECSS-Q-ST-40-12C', ind: 'IEC 61025' },
                      { domain: 'Materials & Processes', nasa: 'NASA-STD-6016C', ecss: 'ECSS-Q-ST-70C Rev.2', ind: 'ASTM, MIL-HDBK-5' },
                      { domain: 'Electrical Bonding', nasa: 'NASA-STD-4003A', ecss: 'ECSS-E-ST-20C Rev.2', ind: 'MIL-STD-464' },
                      { domain: 'Orbital Debris', nasa: 'NASA-STD-8719.17D', ecss: 'ECSS-U-AS-10C Rev.2', ind: 'ISO 24113' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-medium text-[#202124]">{row.domain}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]">
                          {row.nasa.startsWith('NPR') || row.nasa.startsWith('NPD') ? <NprLink code={row.nasa} /> : row.nasa !== '—' ? <NasaDocLink code={row.nasa} /> : <span className="text-[#5f6368]">—</span>}
                        </td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-mono text-xs text-[#3c4043]">{row.ecss}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.ind}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 text-center">
                <Link href="/ecss" className="inline-flex items-center gap-2 text-sm text-[#1a73e8] font-medium hover:underline">
                  <span className="w-5 h-5 bg-[#1a73e8] rounded text-white text-xs flex items-center justify-center font-bold">E</span>
                  Browse the full ECSS Standards Hub →
                </Link>
              </div>
            </section>

            {/* ========== GLOSSARY ========== */}
            <section id="glossary" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Glossary &amp; Abbreviations</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { term: 'CCDev', def: 'Commercial Crew Development' },
                  { term: 'CCtCap', def: 'Commercial Crew Transportation Capability' },
                  { term: 'CDR', def: 'Critical Design Review' },
                  { term: 'CLD', def: 'Commercial LEO Destinations' },
                  { term: 'CLPS', def: 'Commercial Lunar Payload Services' },
                  { term: 'CoFR', def: 'Certification of Flight Readiness' },
                  { term: 'COTS', def: 'Commercial Orbital Transportation Services' },
                  { term: 'CRS', def: 'Commercial Resupply Services' },
                  { term: 'ETA', def: 'Engineering Technical Authority' },
                  { term: 'EVA', def: 'Extravehicular Activity' },
                  { term: 'FRR', def: 'Flight Readiness Review' },
                  { term: 'GNC', def: 'Guidance, Navigation & Control' },
                  { term: 'HLS', def: 'Human Landing System' },
                  { term: 'HRCP', def: 'Human Rating Certification Package' },
                  { term: 'ISS', def: 'International Space Station' },
                  { term: 'IV&V', def: 'Independent Verification & Validation' },
                  { term: 'JSC', def: 'Johnson Space Center' },
                  { term: 'KDP', def: 'Key Decision Point' },
                  { term: 'KSC', def: 'Kennedy Space Center' },
                  { term: 'MSFC', def: 'Marshall Space Flight Center' },
                  { term: 'NODIS', def: 'NASA Online Directives Information System' },
                  { term: 'NPD', def: 'NASA Policy Directive' },
                  { term: 'NPR', def: 'NASA Procedural Requirements' },
                  { term: 'NTSP', def: 'NASA Technical Standards Program' },
                  { term: 'OCE', def: 'Office of the Chief Engineer' },
                  { term: 'OSMA', def: 'Office of Safety & Mission Assurance' },
                  { term: 'PDR', def: 'Preliminary Design Review' },
                  { term: 'PRA', def: 'Probabilistic Risk Assessment' },
                  { term: 'SAA', def: 'Space Act Agreement' },
                  { term: 'SEMP', def: 'Systems Engineering Management Plan' },
                  { term: 'SLS', def: 'Space Launch System' },
                  { term: 'SMA', def: 'Safety & Mission Assurance' },
                  { term: 'SRR', def: 'System Requirements Review' },
                  { term: 'TA', def: 'Technical Authority' },
                  { term: 'WBS', def: 'Work Breakdown Structure' },
                ].map(g => (
                  <div key={g.term} className="bg-[#f8f9fa] border border-[#dadce0] rounded-lg px-4 py-3">
                    <dt className="font-mono font-bold text-[#0b3d91] text-sm">{g.term}</dt>
                    <dd className="text-sm text-[#3c4043] mt-0.5">{g.def}</dd>
                  </div>
                ))}
              </div>
            </section>

            {/* ========== RESOURCES ========== */}
            <section id="resources" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Resources &amp; NASA Centers</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <div className="grid md:grid-cols-2 gap-4 mb-10">
                {[
                  { name: 'NASA Technical Standards System', url: 'https://standards.nasa.gov', desc: 'All NASA-STD, HDBK, SPEC documents + OCE Endorsed Standards' },
                  { name: 'NODIS Library', url: 'https://nodis3.gsfc.nasa.gov', desc: 'All NPDs and NPRs (NASA Directives)' },
                  { name: 'Lessons Learned (LLIS)', url: 'https://llis.nasa.gov', desc: 'Vetted lessons learned from NASA programs and projects' },
                  { name: 'Software Engineering Handbook', url: 'https://swehb.nasa.gov', desc: 'Wiki-based NASA-HDBK-2203 (Software Engineering)' },
                  { name: 'NASA Technical Reports Server (NTRS)', url: 'https://ntrs.nasa.gov', desc: 'NASA Technical Reports — research papers and reports' },
                  { name: 'Space Industry Technical Standards', url: 'https://space.commerce.gov/space-industry-technical-standards/', desc: 'Office of Space Commerce Standards Compendium (2024)' },
                ].map(r => (
                  <a
                    key={r.name}
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-3 p-4 bg-white border border-[#dadce0] rounded-xl hover:border-[#0b3d91] hover:shadow-sm transition-all group"
                  >
                    <div className="w-8 h-8 bg-[#e8eeff] rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4 text-[#0b3d91]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-[#202124] text-sm group-hover:text-[#0b3d91] transition-colors">{r.name}</p>
                      <p className="text-xs text-[#5f6368] mt-0.5">{r.desc}</p>
                    </div>
                  </a>
                ))}
              </div>

              <h3 className="font-bold text-[#202124] mb-4">NASA Centers</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Center</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-16">Code</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Location</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Key Roles</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { name: 'Johnson Space Center', code: 'JSC', loc: 'Houston, TX', roles: 'Human spaceflight, crew systems, mission control, human rating' },
                      { name: 'Kennedy Space Center', code: 'KSC', loc: 'Cape Canaveral, FL', roles: 'Launch operations, Commercial Crew Program' },
                      { name: 'Marshall Space Flight Center', code: 'MSFC', loc: 'Huntsville, AL', roles: 'SLS, propulsion, structural standards' },
                      { name: 'Goddard Space Flight Center', code: 'GSFC', loc: 'Greenbelt, MD', roles: 'Earth science, space science, NODIS host' },
                      { name: 'Jet Propulsion Laboratory', code: 'JPL', loc: 'Pasadena, CA', roles: 'Deep space missions, Mars rovers, FFRDC (Caltech)' },
                      { name: 'Langley Research Center', code: 'LaRC', loc: 'Hampton, VA', roles: 'Aeronautics, atmospheric science, systems analysis' },
                      { name: 'Glenn Research Center', code: 'GRC', loc: 'Cleveland, OH', roles: 'Propulsion, power systems, communications' },
                      { name: 'Ames Research Center', code: 'ARC', loc: 'Moffett Field, CA', roles: 'Computing, thermal protection, astrobiology' },
                      { name: 'Stennis Space Center', code: 'SSC', loc: 'Bay St. Louis, MS', roles: 'Rocket engine testing' },
                      { name: 'Armstrong Flight Research', code: 'AFRC', loc: 'Edwards AFB, CA', roles: 'Flight research and testing' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-medium text-[#202124]">{row.name}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-mono text-[#0b3d91] text-xs">{row.code}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#5f6368]">{row.loc}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.roles}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

          </div>
        </main>
      </div>
    </div>
  );
}
