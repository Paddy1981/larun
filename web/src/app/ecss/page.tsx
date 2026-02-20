'use client';

import { useState } from 'react';
import Link from 'next/link';
import Header from '@/components/Header';

// Helper: generate ECSS document link
function ecssUrl(code: string): string {
  return `https://ecss.nl/?s=${encodeURIComponent(code)}`;
}

function DocLink({ code, children }: { code: string; children?: React.ReactNode }) {
  return (
    <a
      href={ecssUrl(code)}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-[13px] bg-[#e8f0fe] text-[#1a73e8] hover:bg-[#1a73e8] hover:text-white px-2 py-0.5 rounded transition-colors cursor-pointer"
      title={`View ${code} on ecss.nl`}
    >
      {children || code}
      <svg className="w-3 h-3 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </a>
  );
}

const NAV_SECTIONS = [
  { id: 'overview',     label: 'Overview',              icon: '◎' },
  { id: 'branch-m',    label: 'Branch M — Management', icon: 'M' },
  { id: 'branch-e',    label: 'Branch E — Engineering', icon: 'E' },
  { id: 'branch-q',    label: 'Branch Q — Quality',    icon: 'Q' },
  { id: 'branch-u',    label: 'Branch U — Utilities',  icon: 'U' },
  { id: 'branch-spd',  label: 'Branch S / P / D',      icon: '…' },
  { id: 'lifecycle',   label: 'Lifecycle & Reviews',    icon: '↺' },
  { id: 'software',    label: 'Software Standards',     icon: '⌨' },
  { id: 'crossref',    label: 'Cross-Reference',        icon: '⇔' },
  { id: 'glossary',    label: 'Glossary',               icon: 'G' },
  { id: 'resources',   label: 'Resources',              icon: '↗' },
];

export default function EcssPage() {
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

      {/* Mobile sidebar toggle */}
      <button
        className="fixed bottom-6 right-6 z-50 md:hidden bg-[#1a73e8] text-white w-12 h-12 rounded-full shadow-lg flex items-center justify-center"
        onClick={() => setSidebarOpen(!sidebarOpen)}
      >
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className={`fixed top-16 left-0 bottom-0 w-64 bg-[#f8f9fa] border-r border-[#dadce0] z-40 overflow-y-auto transition-transform duration-300 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          {/* Sidebar header */}
          <div className="p-5 border-b border-[#dadce0]">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-[#1a73e8] rounded-lg flex items-center justify-center text-white font-bold text-sm">E</div>
              <div>
                <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider">ECSS</p>
                <p className="text-sm font-bold text-[#202124]">Knowledge Hub</p>
              </div>
            </div>
          </div>

          {/* Stats strip */}
          <div className="flex divide-x divide-[#dadce0] border-b border-[#dadce0]">
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">7</p>
              <p className="text-[10px] text-[#5f6368]">Branches</p>
            </div>
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">139</p>
              <p className="text-[10px] text-[#5f6368]">Standards</p>
            </div>
            <div className="flex-1 py-3 text-center">
              <p className="text-base font-bold text-[#202124]">25K+</p>
              <p className="text-[10px] text-[#5f6368]">Reqs</p>
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
                    ? 'bg-[#e8f0fe] text-[#1a73e8] font-semibold'
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
              href="https://ecss.nl/standards/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-[#1a73e8] hover:underline"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
              </svg>
              Official ECSS Portal ↗
            </a>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 md:ml-64 min-h-screen">
          {/* Page hero */}
          <div className="bg-gradient-to-r from-[#1a73e8] to-[#0d47a1] text-white px-8 py-12">
            <div className="max-w-4xl">
              <div className="inline-flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full text-xs font-medium mb-4">
                European Cooperation for Space Standardization
              </div>
              <h1 className="text-3xl md:text-4xl font-bold mb-3">ECSS Standards Hub</h1>
              <p className="text-white/85 text-lg max-w-2xl">
                The complete reference for all 139 active ECSS standards across 7 branches — covering management, engineering, quality, and sustainability.
              </p>
            </div>
          </div>

          <div className="px-6 md:px-10 py-10 max-w-4xl">

            {/* ========== OVERVIEW ========== */}
            <section id="overview" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Overview</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>

              <p className="text-[#3c4043] mb-6">
                ECSS is the European Cooperation for Space Standardization — a cooperative effort of ESA, national space agencies, and European industry. Its standards provide a unified framework for space project management, engineering, and quality assurance used across all ESA missions and increasingly adopted worldwide.
              </p>

              <div className="grid md:grid-cols-2 gap-4 mb-8">
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Document Hierarchy</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center gap-2">
                      <span className="w-14 text-xs font-mono bg-[#fef3c7] text-[#92400e] px-1.5 py-0.5 rounded text-center">-ST-</span>
                      <span className="text-[#3c4043]">Standards — normative "shall" requirements</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-14 text-xs font-mono bg-[#dbeafe] text-[#1e40af] px-1.5 py-0.5 rounded text-center">-HB-</span>
                      <span className="text-[#3c4043]">Handbooks — informative guidance</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="w-14 text-xs font-mono bg-[#d1fae5] text-[#065f46] px-1.5 py-0.5 rounded text-center">-TM-</span>
                      <span className="text-[#3c4043]">Technical Memoranda — background info</span>
                    </div>
                  </div>
                </div>
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Applicability</h3>
                  <ul className="text-sm text-[#3c4043] space-y-1.5">
                    <li className="flex gap-2"><span className="text-[#1a73e8]">✓</span>All ESA missions and programmes</li>
                    <li className="flex gap-2"><span className="text-[#1a73e8]">✓</span>European national space agencies (CNES, DLR, ASI…)</li>
                    <li className="flex gap-2"><span className="text-[#1a73e8]">✓</span>Prime contractors (Airbus Defence &amp; Space, Thales, OHB…)</li>
                    <li className="flex gap-2"><span className="text-[#1a73e8]">✓</span>Commercial missions opting in</li>
                  </ul>
                </div>
              </div>

              {/* Branch map */}
              <h3 className="font-semibold text-[#202124] mb-4">Branch Structure</h3>
              <div className="grid md:grid-cols-3 gap-3">
                {[
                  { letter: 'M', name: 'Management', color: '#fef3c7', text: '#92400e', desc: 'Project planning, configuration, risk, ILS, cost' },
                  { letter: 'E', name: 'Engineering', color: '#dbeafe', text: '#1e40af', desc: 'Systems, structures, thermal, software, propulsion' },
                  { letter: 'Q', name: 'Quality', color: '#d1fae5', text: '#065f46', desc: 'Product assurance, reliability, safety, EEE, materials' },
                  { letter: 'U', name: 'Utilities', color: '#ede9fe', text: '#4c1d95', desc: 'Space sustainability, spectrum management' },
                  { letter: 'S', name: 'Space Segment', color: '#fee2e2', text: '#991b1b', desc: 'Satellite-specific, launcher-specific standards' },
                  { letter: 'P', name: 'Project', color: '#f0fdf4', text: '#14532d', desc: 'Mission-category tailoring guidelines' },
                  { letter: 'D', name: 'Ground Data Systems', color: '#f5f3ff', text: '#4c1d95', desc: 'Ground segment, mission operations standards' },
                ].map(b => (
                  <div key={b.letter} className="border border-[#dadce0] rounded-lg p-4 hover:border-[#1a73e8] transition-colors">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-7 h-7 rounded flex items-center justify-center font-bold text-sm flex-shrink-0" style={{ background: b.color, color: b.text }}>{b.letter}</span>
                      <span className="font-semibold text-[#202124] text-sm">{b.name}</span>
                    </div>
                    <p className="text-xs text-[#5f6368]">{b.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* ========== BRANCH M ========== */}
            <section id="branch-m" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Branch M — Management</h2>
              <div className="h-1 w-12 bg-[#f59e0b] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Covers project management processes throughout the space project lifecycle, including planning, configuration, risk, cost and logistics.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-48">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-28">Scope</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-M-ST-10C Rev.1', title: 'Project Planning and Implementation', scope: 'Planning' },
                      { code: 'ECSS-M-ST-40C Rev.1', title: 'Configuration and Information Management', scope: 'Config Mgmt' },
                      { code: 'ECSS-M-ST-60C Rev.1', title: 'Cost and Schedule Management', scope: 'Cost/Sched' },
                      { code: 'ECSS-M-ST-70C Rev.1', title: 'Integrated Logistics Support', scope: 'Logistics' },
                      { code: 'ECSS-M-ST-80C Rev.2', title: 'Risk Management', scope: 'Risk' },
                      { code: 'ECSS-M-HB-10A', title: 'Project Planning and Implementation Handbook', scope: 'Guidance' },
                      { code: 'ECSS-M-HB-60A', title: 'Cost Engineering Handbook', scope: 'Cost Guide' },
                      { code: 'ECSS-M-HB-80A', title: 'Risk Management Handbook', scope: 'Risk Guide' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#5f6368]">{row.scope}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-5 bg-[#fef3c7] border border-[#fbbf24] rounded-lg p-4 text-sm text-[#92400e]">
                <strong>Key requirement:</strong> ECSS-M-ST-10C mandates a Project Management Plan (PMP) at project start, covering organisation, WBS, milestones, risk register, and configuration baseline.
              </div>
            </section>

            {/* ========== BRANCH E ========== */}
            <section id="branch-e" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Branch E — Engineering</h2>
              <div className="h-1 w-12 bg-[#3b82f6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">The largest branch — systems engineering, mechanical, thermal, electrical, software, propulsion, communications, and ground systems.</p>

              {/* Systems Engineering sub-group */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-10</span>
                Systems Engineering
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-10C Rev.1', title: 'System Engineering General Requirements' },
                      { code: 'ECSS-E-ST-10-02C Rev.1', title: 'Verification' },
                      { code: 'ECSS-E-ST-10-03C Rev.1', title: 'Testing' },
                      { code: 'ECSS-E-ST-10-04C Rev.1', title: 'Space Environment' },
                      { code: 'ECSS-E-ST-10-06C', title: 'Technical Requirements Specification' },
                      { code: 'ECSS-E-ST-10-09C', title: 'Interface Management' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Structural sub-group */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-32</span>
                Structural Engineering
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-32C Rev.1', title: 'Structural General Requirements' },
                      { code: 'ECSS-E-ST-32-01C Rev.2', title: 'Fracture Control' },
                      { code: 'ECSS-E-ST-32-02C Rev.1', title: 'Structural Design and Verification of Pressurised Hardware' },
                      { code: 'ECSS-E-ST-32-08C Rev.2', title: 'Fasteners' },
                      { code: 'ECSS-E-ST-32-10C Rev.1', title: 'Structural Factors of Safety for Spaceflight Hardware' },
                      { code: 'ECSS-E-HB-32-20A', title: 'Structural Materials Handbook' },
                      { code: 'ECSS-E-HB-32-26A', title: 'Spacecraft Mechanical Loads Analysis Handbook' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Electrical */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-20</span>
                Electrical &amp; Electronic
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-20C Rev.2', title: 'Electrical and Electronic' },
                      { code: 'ECSS-E-ST-20-07C Rev.1', title: 'Electromagnetic Compatibility (EMC)' },
                      { code: 'ECSS-E-ST-20-08C Rev.1', title: 'Photovoltaic Assemblies and Components' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Thermal */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-31</span>
                Thermal Control
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-31C Rev.1', title: 'Thermal Control for Space Vehicles' },
                      { code: 'ECSS-E-HB-31-01A', title: 'Thermal Design Handbook' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Propulsion */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-33/35</span>
                Propulsion
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-33-01C Rev.2', title: 'Liquid Propulsion for Spacecraft' },
                      { code: 'ECSS-E-ST-33-11C Rev.1', title: 'Electric Propulsion' },
                      { code: 'ECSS-E-ST-35C Rev.1', title: 'Propulsion General' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Software */}
              <h3 className="font-semibold text-[#202124] mb-3 flex items-center gap-2">
                <span className="bg-[#dbeafe] text-[#1e40af] text-xs px-2 py-0.5 rounded font-mono">E-40</span>
                Software Engineering
              </h3>
              <div className="overflow-x-auto mb-8">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-E-ST-40C Rev.1', title: 'Software Engineering General Requirements' },
                      { code: 'ECSS-E-HB-40A', title: 'Software Engineering Handbook' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ========== BRANCH Q ========== */}
            <section id="branch-q" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Branch Q — Quality Assurance</h2>
              <div className="h-1 w-12 bg-[#10b981] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Product assurance, reliability, FMECA, safety analysis, EEE components, materials, and software product assurance.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-28">Domain</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-Q-ST-10C Rev.1', title: 'Product Assurance Management', domain: 'PA Mgmt' },
                      { code: 'ECSS-Q-ST-20C Rev.1', title: 'Quality Assurance', domain: 'QA' },
                      { code: 'ECSS-Q-ST-20-07C', title: 'Quality Standard for Complex EEE Components', domain: 'EEE' },
                      { code: 'ECSS-Q-ST-20-08C', title: 'Measurement Uncertainty', domain: 'Measurement' },
                      { code: 'ECSS-Q-ST-30C Rev.1', title: 'Dependability', domain: 'Reliability' },
                      { code: 'ECSS-Q-ST-30-02C', title: 'Failure Mode Effects Criticality Analysis (FMECA)', domain: 'Safety' },
                      { code: 'ECSS-Q-ST-40C Rev.1', title: 'Safety', domain: 'Safety' },
                      { code: 'ECSS-Q-ST-40-12C', title: 'Fault Tree Analysis (FTA)', domain: 'Safety' },
                      { code: 'ECSS-Q-ST-60C Rev.2', title: 'EEE Components', domain: 'EEE' },
                      { code: 'ECSS-Q-ST-60-02C Rev.2', title: 'COTS EEE Components', domain: 'EEE COTS' },
                      { code: 'ECSS-Q-ST-70C Rev.2', title: 'Materials, Mechanical Parts, and Processes', domain: 'Materials' },
                      { code: 'ECSS-Q-ST-70-01C Rev.1', title: 'Cleanliness and Contamination Control', domain: 'Cleanliness' },
                      { code: 'ECSS-Q-ST-70-04C', title: 'Thermal Testing for Soldered Connections', domain: 'Soldering' },
                      { code: 'ECSS-Q-ST-70-12C', title: 'Design Rules for PCBs', domain: 'PCB' },
                      { code: 'ECSS-Q-ST-80C Rev.1', title: 'Software Product Assurance', domain: 'SW PA' },
                      { code: 'ECSS-Q-HB-30-01A', title: 'Dependability Handbook', domain: 'Guidance' },
                      { code: 'ECSS-Q-HB-80-04A', title: 'Software Metrication Handbook', domain: 'SW Guidance' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#5f6368] text-xs">{row.domain}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-5 bg-[#d1fae5] border border-[#6ee7b7] rounded-lg p-4 text-sm text-[#065f46]">
                <strong>FMECA note:</strong> ECSS-Q-ST-30-02C requires FMECA at both functional and hardware levels. Critical items (single-point failures) must be tracked in a FRACAS (Failure Reporting, Analysis, and Corrective Action System).
              </div>
            </section>

            {/* ========== BRANCH U ========== */}
            <section id="branch-u" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Branch U — Utilities / Sustainability</h2>
              <div className="h-1 w-12 bg-[#8b5cf6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Cross-cutting standards for space sustainability, orbital debris mitigation, and spectrum management.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { code: 'ECSS-U-AS-10C Rev.2', title: 'Space Sustainability (Orbital Debris Mitigation)' },
                      { code: 'ECSS-U-ST-10C', title: 'Space Sustainability — Legal and Regulatory Framework' },
                      { code: 'ECSS-U-ST-20C', title: 'Electromagnetic Spectrum Management' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.code} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-5 bg-[#ede9fe] border border-[#c4b5fd] rounded-lg p-4 text-sm text-[#4c1d95]">
                <strong>Debris mitigation:</strong> ECSS-U-AS-10C aligns with ISO 24113 and IADC guidelines. Key requirements include post-mission disposal within 25 years for LEO, and passivation of energy sources within 5 years of mission end.
              </div>
            </section>

            {/* ========== BRANCH S/P/D ========== */}
            <section id="branch-spd" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Branch S / P / D</h2>
              <div className="h-1 w-12 bg-[#ef4444] rounded mb-6"></div>

              <div className="grid md:grid-cols-3 gap-5">
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-7 h-7 rounded bg-[#fee2e2] text-[#991b1b] flex items-center justify-center font-bold text-sm">S</span>
                    <h3 className="font-semibold text-[#202124]">Space Segment</h3>
                  </div>
                  <p className="text-sm text-[#5f6368] mb-3">Spacecraft and launcher-specific standards. Includes satellite bus, payload integration, and launch vehicle interfaces.</p>
                  <ul className="space-y-1.5 text-xs">
                    <li><DocLink code="ECSS-S-ST-00C" /></li>
                    <li className="text-[#5f6368]">Tailoring guidelines for satellite/launcher categories</li>
                  </ul>
                </div>

                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-7 h-7 rounded bg-[#f0fdf4] text-[#14532d] flex items-center justify-center font-bold text-sm">P</span>
                    <h3 className="font-semibold text-[#202124]">Project</h3>
                  </div>
                  <p className="text-sm text-[#5f6368] mb-3">Mission-category tailoring framework. Defines compliance levels for different mission classes (A–D).</p>
                  <ul className="space-y-1.5 text-xs">
                    <li><DocLink code="ECSS-P-ST-00A Rev.1" /></li>
                    <li className="text-[#5f6368]">Framework for mission-class tailoring</li>
                  </ul>
                </div>

                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <span className="w-7 h-7 rounded bg-[#f5f3ff] text-[#4c1d95] flex items-center justify-center font-bold text-sm">D</span>
                    <h3 className="font-semibold text-[#202124]">Ground Data Systems</h3>
                  </div>
                  <p className="text-sm text-[#5f6368] mb-3">Ground segment, mission data management, and telemetry standards.</p>
                  <ul className="space-y-1.5 text-xs">
                    <li><DocLink code="ECSS-E-ST-65C Rev.1" /></li>
                    <li className="text-[#5f6368]">Ground Systems and Operations</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* ========== LIFECYCLE ========== */}
            <section id="lifecycle" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Lifecycle &amp; Reviews</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">ECSS defines a phased project lifecycle with mandatory reviews at each phase gate, aligned with ESA's project phases 0/A/B/C/D/E/F.</p>

              <div className="space-y-3 mb-8">
                {[
                  { phase: '0', name: 'Mission Analysis / Needs Identification', reviews: 'MDR' },
                  { phase: 'A', name: 'Feasibility', reviews: 'PRR, SRR' },
                  { phase: 'B', name: 'Preliminary Definition', reviews: 'PDR' },
                  { phase: 'C', name: 'Detailed Definition', reviews: 'CDR' },
                  { phase: 'D', name: 'Qualification and Production', reviews: 'QR, AR' },
                  { phase: 'E', name: 'Utilisation', reviews: 'ORR, FRR, LRR, CRR, ELR' },
                  { phase: 'F', name: 'Disposal', reviews: 'MCR' },
                ].map(p => (
                  <div key={p.phase} className="flex items-center gap-4 p-4 bg-[#f8f9fa] border border-[#dadce0] rounded-lg">
                    <span className="w-8 h-8 bg-[#1a73e8] text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">{p.phase}</span>
                    <div className="flex-1">
                      <p className="font-medium text-[#202124] text-sm">{p.name}</p>
                    </div>
                    <span className="text-xs font-mono text-[#5f6368] bg-white border border-[#dadce0] px-2 py-1 rounded">{p.reviews}</span>
                  </div>
                ))}
              </div>

              <div className="bg-[#e8f0fe] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e40af]">
                <strong>Review types:</strong> MDR (Mission Definition Review), PRR (Preliminary Requirements Review), SRR (System Requirements Review), PDR (Preliminary Design Review), CDR (Critical Design Review), QR (Qualification Review), AR (Acceptance Review), ORR (Operational Readiness Review), FRR (Flight Readiness Review).
              </div>
            </section>

            {/* ========== SOFTWARE ========== */}
            <section id="software" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Software Standards</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">ECSS defines software engineering and assurance requirements across two branches: E-40 (Engineering) and Q-80 (Quality). Together they govern the full software lifecycle for space projects.</p>

              <div className="grid md:grid-cols-2 gap-5 mb-8">
                <div className="border border-[#dadce0] rounded-xl overflow-hidden">
                  <div className="bg-[#dbeafe] px-4 py-3">
                    <h3 className="font-semibold text-[#1e40af] text-sm"><DocLink code="ECSS-E-ST-40C Rev.1" /> — Engineering</h3>
                  </div>
                  <div className="p-4 text-sm text-[#3c4043] space-y-1.5">
                    <p>• Software development process requirements</p>
                    <p>• Documentation requirements (SRS, SDD, ICD, STS, STR)</p>
                    <p>• Verification and validation at each phase</p>
                    <p>• Software classification (levels A–E by criticality)</p>
                  </div>
                </div>
                <div className="border border-[#dadce0] rounded-xl overflow-hidden">
                  <div className="bg-[#d1fae5] px-4 py-3">
                    <h3 className="font-semibold text-[#065f46] text-sm"><DocLink code="ECSS-Q-ST-80C Rev.1" /> — Product Assurance</h3>
                  </div>
                  <div className="p-4 text-sm text-[#3c4043] space-y-1.5">
                    <p>• Software PA plan and reviews</p>
                    <p>• Non-conformance and change control</p>
                    <p>• Supplier audits and qualification</p>
                    <p>• Metrics collection and analysis</p>
                  </div>
                </div>
              </div>

              <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                <h3 className="font-semibold text-[#202124] mb-3">Software Classification (ECSS-E-ST-40)</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="bg-white text-left">
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Level</th>
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Criticality</th>
                        <th className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">Example</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        { level: 'A', crit: 'Mission-critical / Safety-critical', ex: 'Attitude control, life-support' },
                        { level: 'B', crit: 'Mission-critical, not safety', ex: 'Payload data handling' },
                        { level: 'C', crit: 'Not mission-critical', ex: 'Ground support tools' },
                        { level: 'D', crit: 'Non-mission software', ex: 'Admin, documentation tools' },
                        { level: 'E', crit: 'COTS / reused (no source)', ex: 'OS, RTOS' },
                      ].map((row, i) => (
                        <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] font-bold text-[#1a73e8]">{row.level}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#3c4043]">{row.crit}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#5f6368]">{row.ex}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* ========== CROSS-REFERENCE ========== */}
            <section id="crossref" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Cross-Reference: ECSS ↔ NASA ↔ Industry</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">Mapping between ECSS standards, NASA standards, and key industry/military equivalents.</p>

              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Domain</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">ECSS</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">NASA</th>
                      <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Industry</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { domain: 'Systems Engineering', ecss: 'ECSS-E-ST-10C Rev.1', nasa: 'NPR 7123.1C', ind: 'ISO/IEC 15288' },
                      { domain: 'Program Management', ecss: 'ECSS-M-ST-10C Rev.1', nasa: 'NPR 7120.5F', ind: 'PMI PMBOK' },
                      { domain: 'Software Engineering', ecss: 'ECSS-E-ST-40C Rev.1', nasa: 'NPR 7150.2C', ind: 'DO-178C, IEEE 12207' },
                      { domain: 'Software Assurance', ecss: 'ECSS-Q-ST-80C Rev.1', nasa: 'NASA-STD-8739.8B', ind: 'DO-178C, IEC 61508-3' },
                      { domain: 'Structural Design', ecss: 'ECSS-E-ST-32C Rev.1', nasa: 'NASA-STD-5001B', ind: 'AIAA S-110, S-111' },
                      { domain: 'Fracture Control', ecss: 'ECSS-E-ST-32-01C Rev.2', nasa: 'NASA-STD-5019A', ind: 'MIL-STD-1530' },
                      { domain: 'Safety / FMEA', ecss: 'ECSS-Q-ST-40C Rev.1', nasa: 'NPR 8715.3C', ind: 'MIL-STD-882E, IEC 61508' },
                      { domain: 'Fault Tree Analysis', ecss: 'ECSS-Q-ST-40-12C', nasa: 'NASA Fault Tree HB', ind: 'IEC 61025' },
                      { domain: 'Materials & Processes', ecss: 'ECSS-Q-ST-70C Rev.2', nasa: 'NASA-STD-6016C', ind: 'ASTM, MIL-HDBK-5' },
                      { domain: 'Electrical Bonding', ecss: 'ECSS-E-ST-20C Rev.2', nasa: 'NASA-STD-4003A', ind: 'MIL-STD-464' },
                      { domain: 'Orbital Debris', ecss: 'ECSS-U-AS-10C Rev.2', nasa: 'NASA-STD-8719.17D', ind: 'ISO 24113' },
                      { domain: 'EMC', ecss: 'ECSS-E-ST-20-07C', nasa: 'Center standards', ind: 'MIL-STD-461G' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-medium text-[#202124]">{row.domain}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4]"><DocLink code={row.ecss} /></td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] font-mono text-xs text-[#5f6368]">{row.nasa}</td>
                        <td className="px-4 py-3 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.ind}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-4 text-center">
                <Link href="/nasa" className="inline-flex items-center gap-2 text-sm text-[#0b3d91] font-medium hover:underline">
                  <span className="w-5 h-5 bg-[#0b3d91] rounded text-white text-xs flex items-center justify-center font-bold">N</span>
                  Browse the full NASA Standards Hub →
                </Link>
              </div>
            </section>

            {/* ========== GLOSSARY ========== */}
            <section id="glossary" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Glossary</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                {[
                  { term: 'CDR', def: 'Critical Design Review' },
                  { term: 'CDF', def: 'Concurrent Design Facility' },
                  { term: 'DRD', def: 'Document Requirements Definition' },
                  { term: 'EEE', def: 'Electrical, Electronic & Electromechanical' },
                  { term: 'FMECA', def: 'Failure Mode, Effects & Criticality Analysis' },
                  { term: 'FRACAS', def: 'Failure Reporting, Analysis & Corrective Action System' },
                  { term: 'FRR', def: 'Flight Readiness Review' },
                  { term: 'FTA', def: 'Fault Tree Analysis' },
                  { term: 'GSE', def: 'Ground Support Equipment' },
                  { term: 'ICD', def: 'Interface Control Document' },
                  { term: 'ILS', def: 'Integrated Logistics Support' },
                  { term: 'KDP', def: 'Key Decision Point' },
                  { term: 'MCR', def: 'Mission Close-out Review' },
                  { term: 'MDR', def: 'Mission Definition Review' },
                  { term: 'NCR', def: 'Non-Conformance Report' },
                  { term: 'ORR', def: 'Operational Readiness Review' },
                  { term: 'PA', def: 'Product Assurance' },
                  { term: 'PDR', def: 'Preliminary Design Review' },
                  { term: 'PMP', def: 'Project Management Plan' },
                  { term: 'PRR', def: 'Preliminary Requirements Review' },
                  { term: 'RAMS', def: 'Reliability, Availability, Maintainability & Safety' },
                  { term: 'RID', def: 'Review Item Discrepancy' },
                  { term: 'SE', def: 'Systems Engineering' },
                  { term: 'SEMP', def: 'Systems Engineering Management Plan' },
                  { term: 'SRR', def: 'System Requirements Review' },
                  { term: 'STR', def: 'Software Test Report' },
                  { term: 'TRL', def: 'Technology Readiness Level' },
                  { term: 'V&V', def: 'Verification & Validation' },
                  { term: 'WBS', def: 'Work Breakdown Structure' },
                ].map(g => (
                  <div key={g.term} className="bg-[#f8f9fa] border border-[#dadce0] rounded-lg px-4 py-3">
                    <dt className="font-mono font-bold text-[#1a73e8] text-sm">{g.term}</dt>
                    <dd className="text-sm text-[#3c4043] mt-0.5">{g.def}</dd>
                  </div>
                ))}
              </div>
            </section>

            {/* ========== RESOURCES ========== */}
            <section id="resources" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Resources</h2>
              <div className="h-1 w-12 bg-[#1a73e8] rounded mb-6"></div>

              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { name: 'ECSS Standards Portal', url: 'https://ecss.nl/standards/', desc: 'Official source for all ECSS standards — search, filter by branch, download PDFs' },
                  { name: 'ESA Engineering Standards', url: 'https://www.esa.int/Enabling_Support/Space_Engineering_Technology/Engineering_Standards', desc: 'ESA\'s overview of ECSS adoption and project requirements' },
                  { name: 'ECSS Secretariat', url: 'https://ecss.nl/', desc: 'ECSS organisation homepage — membership, working groups, news' },
                  { name: 'NASA Standards (Cross-ref)', url: 'https://standards.nasa.gov', desc: 'NASA\'s equivalent technical standards program for comparison' },
                  { name: 'ISO Space Standards', url: 'https://www.iso.org/committee/46694.html', desc: 'ISO/TC 20/SC 14 — Space Systems standards committee' },
                  { name: 'ECSS Glossary', url: 'https://ecss.nl/glossary/', desc: 'Searchable ECSS glossary of space terms and abbreviations' },
                ].map(r => (
                  <a
                    key={r.name}
                    href={r.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-start gap-3 p-4 bg-white border border-[#dadce0] rounded-xl hover:border-[#1a73e8] hover:shadow-sm transition-all group"
                  >
                    <div className="w-8 h-8 bg-[#e8f0fe] rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5">
                      <svg className="w-4 h-4 text-[#1a73e8]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
                      </svg>
                    </div>
                    <div>
                      <p className="font-medium text-[#202124] text-sm group-hover:text-[#1a73e8] transition-colors">{r.name}</p>
                      <p className="text-xs text-[#5f6368] mt-0.5">{r.desc}</p>
                    </div>
                  </a>
                ))}
              </div>
            </section>

          </div>
        </main>
      </div>
    </div>
  );
}
