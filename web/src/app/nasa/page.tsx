'use client';

import { useState } from 'react';
import Link from 'next/link';
import Header from '@/components/Header';

function nasaStdUrl(code: string): string {
  const c = code.toLowerCase().replace(/[.\s]/g, '-').replace(/[^a-z0-9-]/g, '');
  if (code.startsWith('NPR') || code.startsWith('NPD')) return 'https://nodis3.gsfc.nasa.gov/';
  return `https://standards.nasa.gov/standard/nasa/${c}`;
}

function NasaDocLink({ code, children }: { code: string; children?: React.ReactNode }) {
  return (
    <a href={nasaStdUrl(code)} target="_blank" rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-[13px] bg-[#e8eeff] text-[#0b3d91] hover:bg-[#0b3d91] hover:text-white px-2 py-0.5 rounded transition-colors cursor-pointer"
      title={`View ${code}`}>
      {children || code}
      <svg className="w-3 h-3 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </a>
  );
}

function NprLink({ code, children }: { code: string; children?: React.ReactNode }) {
  return (
    <a href="https://nodis3.gsfc.nasa.gov/" target="_blank" rel="noopener noreferrer"
      className="inline-flex items-center gap-1 font-mono text-[13px] bg-[#fef3c7] text-[#92400e] hover:bg-[#92400e] hover:text-white px-2 py-0.5 rounded transition-colors cursor-pointer"
      title={`View ${code} on NODIS`}>
      {children || code}
      <svg className="w-3 h-3 opacity-70" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
      </svg>
    </a>
  );
}

const NAV_SECTIONS = [
  { id: 'overview',      label: 'Overview',                icon: '◎',  group: 'Overview' },
  { id: 'governance',    label: 'Governance Framework',    icon: '⚖',  group: 'Overview' },
  { id: 'lifecycle',     label: 'Lifecycle & Reviews',     icon: '↺',  group: 'Overview' },
  { id: 'syseng',        label: '1000 — Systems Eng.',     icon: '⬡',  group: 'Technical Standards' },
  { id: 'comms',         label: '2000 — Comms & IT',       icon: '📡', group: 'Technical Standards' },
  { id: 'human',         label: '3000 — Human Systems',    icon: '👤', group: 'Technical Standards' },
  { id: 'electrical',    label: '4000 — Electrical',       icon: '⚡', group: 'Technical Standards' },
  { id: 'structures',    label: '5000 — Structures',       icon: '▦',  group: 'Technical Standards' },
  { id: 'materials',     label: '6000 — Materials',        icon: '◈',  group: 'Technical Standards' },
  { id: 'test-env',      label: '7000 — Test & Env',       icon: '⊙',  group: 'Technical Standards' },
  { id: 'safety',        label: '8000 — Safety & SMA',     icon: '⚠',  group: 'Technical Standards' },
  { id: 'facilities',    label: '10000 — Facilities',      icon: '🏗', group: 'Technical Standards' },
  { id: 'software',      label: 'Software (Cross-Cut)',    icon: '⌨',  group: 'Technical Standards' },
  { id: 'commercial',    label: 'Commercial Programs',     icon: '🚀', group: 'Commercial Space' },
  { id: 'cots-crs',      label: 'COTS & CRS (Cargo)',      icon: '📦', group: 'Commercial Space' },
  { id: 'ccp',           label: 'Commercial Crew (CCP)',   icon: '👨‍🚀', group: 'Commercial Space' },
  { id: 'clps',          label: 'CLPS (Lunar)',            icon: '🌑', group: 'Commercial Space' },
  { id: 'cld',           label: 'CLD (Stations)',          icon: '🛸', group: 'Commercial Space' },
  { id: 'hls',           label: 'HLS (Lunar Landers)',     icon: '🌙', group: 'Commercial Space' },
  { id: 'artemis',       label: 'Artemis Architecture',    icon: '🔭', group: 'Commercial Space' },
  { id: 'complete-list', label: 'Complete List',           icon: '≡',  group: 'Reference' },
  { id: 'npr-index',     label: 'NPR / NPD Index',         icon: 'P',  group: 'Reference' },
  { id: 'crossref',      label: 'Cross-Reference',         icon: '⇔',  group: 'Reference' },
  { id: 'glossary',      label: 'Glossary',                icon: 'G',  group: 'Reference' },
  { id: 'resources',     label: 'Resources & Centers',     icon: '↗',  group: 'Reference' },
];

const GROUPS = ['Overview', 'Technical Standards', 'Commercial Space', 'Reference'];

function StdTable({ rows, showDate = false }: { rows: { code: string; title: string; date?: string; type?: string }[]; showDate?: boolean }) {
  return (
    <div className="overflow-x-auto mb-6">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="bg-[#f8f9fa] text-left">
            <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-52">Standard</th>
            <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0]">Title</th>
            {showDate && <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-36">Date</th>}
            {rows.some(r => r.type) && <th className="px-4 py-3 font-semibold text-[#202124] border-b border-[#dadce0] w-20">Type</th>}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
              <td className="px-4 py-3 border-b border-[#f1f3f4]">
                {row.code.startsWith('NPR') || row.code.startsWith('NPD') ? <NprLink code={row.code} /> : <NasaDocLink code={row.code} />}
              </td>
              <td className="px-4 py-3 border-b border-[#f1f3f4] text-[#3c4043]">{row.title}</td>
              {showDate && <td className="px-4 py-3 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.date || '—'}</td>}
              {row.type && (
                <td className="px-4 py-3 border-b border-[#f1f3f4]">
                  <span className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                    row.type === 'STD' ? 'bg-[#dbeafe] text-[#1e40af]' :
                    row.type === 'HDBK' ? 'bg-[#fef3c7] text-[#92400e]' :
                    row.type === 'SPEC' ? 'bg-[#fee2e2] text-[#991b1b]' :
                    'bg-[#d1fae5] text-[#065f46]'
                  }`}>{row.type}</span>
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

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

      {/* Mobile FAB */}
      <button
        className="fixed bottom-6 right-6 z-50 md:hidden bg-[#0b3d91] text-white w-12 h-12 rounded-full shadow-lg flex items-center justify-center"
        onClick={() => setSidebarOpen(!sidebarOpen)}>
        <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16"/>
        </svg>
      </button>

      <div className="flex pt-16">
        {/* Sidebar */}
        <aside className={`fixed top-16 left-0 bottom-0 w-64 bg-[#f8f9fa] border-r border-[#dadce0] z-40 overflow-y-auto transition-transform duration-300 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0`}>
          <div className="p-5 border-b border-[#dadce0]">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 bg-[#0b3d91] rounded-lg flex items-center justify-center text-white font-bold text-sm">N</div>
              <div>
                <p className="text-xs font-semibold text-[#5f6368] uppercase tracking-wider">NASA</p>
                <p className="text-sm font-bold text-[#202124]">Standards Hub</p>
              </div>
            </div>
          </div>

          <div className="flex divide-x divide-[#dadce0] border-b border-[#dadce0]">
            {[['83', 'Documents'], ['9', 'Disciplines'], ['7', 'Programs']].map(([v, l]) => (
              <div key={l} className="flex-1 py-3 text-center">
                <p className="text-base font-bold text-[#202124]">{v}</p>
                <p className="text-[10px] text-[#5f6368]">{l}</p>
              </div>
            ))}
          </div>

          <nav className="p-3">
            {GROUPS.map(group => (
              <div key={group}>
                <p className="text-[10px] font-semibold text-[#5f6368] uppercase tracking-wider px-3 pt-4 pb-1">{group}</p>
                {NAV_SECTIONS.filter(s => s.group === group).map(s => (
                  <button key={s.id} onClick={() => scrollTo(s.id)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-left transition-colors mb-0.5 ${
                      activeSection === s.id ? 'bg-[#e8eeff] text-[#0b3d91] font-semibold' : 'text-[#3c4043] hover:bg-[#f1f3f4]'
                    }`}>
                    <span className="w-5 h-5 rounded text-center leading-5 text-xs font-mono flex-shrink-0">{s.icon}</span>
                    <span className="text-xs">{s.label}</span>
                  </button>
                ))}
              </div>
            ))}
          </nav>

          <div className="p-4 border-t border-[#dadce0] mt-2">
            <a href="https://standards.nasa.gov" target="_blank" rel="noopener noreferrer"
              className="flex items-center gap-2 text-xs text-[#0b3d91] hover:underline">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"/>
              </svg>
              standards.nasa.gov ↗
            </a>
          </div>
        </aside>

        {/* Main */}
        <main className="flex-1 md:ml-64 min-h-screen">
          {/* Hero */}
          <div className="bg-gradient-to-r from-[#0b3d91] to-[#1a237e] text-white px-8 py-12">
            <div className="max-w-4xl">
              <div className="inline-flex items-center gap-2 bg-white/20 px-3 py-1 rounded-full text-xs font-medium mb-4">
                NASA Technical Standards Program (NTSP)
              </div>
              <h1 className="text-3xl md:text-4xl font-bold mb-3">NASA Standards Hub</h1>
              <p className="text-white/85 text-lg max-w-2xl">
                Complete reference for 83 NASA-STD, NASA-HDBK, and NASA-SPEC documents across 9 engineering discipline categories — plus commercial space programs, Artemis architecture, and governance framework.
              </p>
              <div className="flex flex-wrap gap-3 mt-6">
                {['NASA-STD (≈45)', 'NASA-HDBK (≈37)', 'NASA-SPEC (1)', '7 Commercial Programs', '60+ ASTM Cross-Refs'].map(t => (
                  <span key={t} className="bg-white/15 text-white text-xs px-3 py-1.5 rounded-full font-medium">{t}</span>
                ))}
              </div>
            </div>
          </div>

          <div className="px-6 md:px-10 py-10 max-w-4xl">

            {/* ===== OVERVIEW ===== */}
            <section id="overview" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Overview</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-6">
                The NASA Technical Standards Program (NTSP) maintains the Agency&apos;s collection of technical standards, handbooks, and specifications. Documents are accessed via the NASA Technical Standards System (NTSS) at <a href="https://standards.nasa.gov" target="_blank" rel="noopener noreferrer" className="text-[#0b3d91] hover:underline">standards.nasa.gov</a>. Directives (NPRs/NPDs) live in NODIS.
              </p>
              <div className="grid md:grid-cols-3 gap-4 mb-6">
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Document Types</h3>
                  <div className="space-y-2 text-sm">
                    {[
                      { badge: 'NASA-STD', bg: '#e8eeff', color: '#0b3d91', desc: 'Mandatory "shall" requirements' },
                      { badge: 'NASA-HDBK', bg: '#fef3c7', color: '#92400e', desc: 'Guidance "should" recommendations' },
                      { badge: 'NPR/NPD', bg: '#d1fae5', color: '#065f46', desc: 'Procedural requirements & policy' },
                      { badge: 'NASA-SPEC', bg: '#fee2e2', color: '#991b1b', desc: 'Product specifications' },
                    ].map(d => (
                      <div key={d.badge} className="flex items-center gap-2">
                        <span className="w-20 text-xs font-mono px-1.5 py-0.5 rounded text-center" style={{ background: d.bg, color: d.color }}>{d.badge}</span>
                        <span className="text-[#3c4043] text-xs">{d.desc}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Number Series</h3>
                  <div className="space-y-1.5 text-xs font-mono text-[#3c4043]">
                    {[['1000–1999','Systems Engineering'],['2000–2999','Comms & IT'],['3000–3999','Human Systems'],['4000–4999','Electrical'],['5000–5999','Structures/Mech'],['6000–6999','Materials'],['7000–7999','Test & Environment'],['8000–8999','Safety & SMA'],['10000+','Facilities']].map(([range, label]) => (
                      <p key={range}><span className="text-[#0b3d91] font-bold">{range}</span> {label}</p>
                    ))}
                  </div>
                </div>
                <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                  <h3 className="font-semibold text-[#202124] mb-3">Key Facts</h3>
                  <ul className="text-xs text-[#3c4043] space-y-1.5">
                    <li>✓ Program established 1997 by OCE</li>
                    <li>✓ 83 active technical standards</li>
                    <li>✓ 10 NASA Centers + JPL (FFRDC)</li>
                    <li>✓ OMB A-119: prefer industry stds</li>
                    <li>✓ &quot;Internet Public&quot; = freely downloadable</li>
                    <li>✓ Feedback via NTSS portal</li>
                  </ul>
                </div>
              </div>
            </section>

            {/* ===== GOVERNANCE ===== */}
            <section id="governance" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Governance Framework</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <h3 className="font-semibold text-[#202124] mb-3">Document Hierarchy (8 Levels)</h3>
              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      {['Level','Document Type','Prefix','Nature','Example'].map(h => (
                        <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['1 — Federal','U.S. Code / OMB Circulars','—','Law / Regulation','51 U.S.C. § 20113; OMB A-119'],
                      ['2 — Policy','NASA Policy Directive','NPD','What (Policy)','NPD 7120.4 — Engineering & PM Policy'],
                      ['3 — Procedure','NASA Procedural Requirements','NPR','How (Requirements)','NPR 7120.5F — Space Flight PM'],
                      ['4 — Standard','NASA Technical Standard','NASA-STD-','Mandatory ("shall")','NASA-STD-5019A — Fracture Control'],
                      ['4 — Spec','NASA Specification','NASA-SPEC-','Mandatory ("shall")','NASA-SPEC-5022 — Pyrovalves'],
                      ['4 — Handbook','NASA Technical Handbook','NASA-HDBK-','Guidance ("should")','NASA-HDBK-2203 — Software Eng.'],
                      ['5 — Center','Center Standards / CPR / CPD','Varies','Center-specific','JPL Design Principles, MSFC-SPEC-'],
                      ['6 — Endorsed','OCE Endorsed Standards','Varies','Pick-list for programs','AIAA, SAE, IEEE, ASTM, MIL-STD'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        {row.map((cell, j) => (
                          <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">Three Technical Authorities</h3>
              <div className="bg-[#e8eeff] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e3a8a] mb-6">
                <p><strong>Engineering Technical Authority (ETA)</strong> — Responsible for engineering design processes, specifications, rules, and best practices. Delegated by Center Directors.</p>
                <p className="mt-2"><strong>Safety & Mission Assurance Technical Authority (SMA TA)</strong> — Independent safety oversight managed per NASA-STD-8709.20.</p>
                <p className="mt-2"><strong>Health & Medical Technical Authority (HMTA)</strong> — Crew health/medical requirements compliance for human spaceflight.</p>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                {[
                  { code: 'NPR 7120.5F', title: 'Space Flight Program & Project Management', desc: 'Lifecycle phases Pre-A through F, KDPs, life-cycle reviews, product requirements.' },
                  { code: 'NPR 7123.1C', title: 'Systems Engineering Processes & Requirements', desc: '17 SE processes covering technical requirements, design, integration, and V&V.' },
                  { code: 'NPR 7150.2C', title: 'Software Engineering Requirements', desc: 'Mandatory software engineering, classification levels A–E, IV&V requirements.' },
                  { code: 'NPR 8000.4', title: 'Agency Risk Management', desc: 'Risk identification, analysis, handling, and monitoring across all programs.' },
                ].map(item => (
                  <div key={item.code} className="border border-[#dadce0] rounded-xl p-4 hover:border-[#0b3d91] transition-colors">
                    <div className="mb-2"><NprLink code={item.code} /></div>
                    <h4 className="font-semibold text-[#202124] text-sm mb-1">{item.title}</h4>
                    <p className="text-xs text-[#5f6368]">{item.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* ===== LIFECYCLE ===== */}
            <section id="lifecycle" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Lifecycle &amp; Reviews</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Governed by <NprLink code="NPR 7120.5F" />. Phases Pre-A through B = <strong>Formulation</strong>; C through F = <strong>Implementation</strong>.</p>
              <div className="flex flex-wrap gap-2 mb-6">
                {[
                  { phase: 'Pre-A', name: 'Concept Studies', review: '→ KDP A', bg: '#dbeafe' },
                  { phase: 'A', name: 'Concept & Tech Dev', review: 'MCR/SRR → KDP B', bg: '#dbeafe' },
                  { phase: 'B', name: 'Preliminary Design', review: 'SDR/PDR → KDP C', bg: '#e0e7ff' },
                  { phase: 'C', name: 'Final Design & Fab', review: 'CDR → KDP D', bg: '#ede9fe' },
                  { phase: 'D', name: 'Assembly, I&T', review: 'SIR/ORR → KDP E', bg: '#fef3c7' },
                  { phase: 'E', name: 'Operations', review: 'FRR/DR → KDP F', bg: '#d1fae5' },
                  { phase: 'F', name: 'Closeout', review: 'DR', bg: '#fee2e2' },
                ].map(p => (
                  <div key={p.phase} className="flex-1 min-w-[90px] border border-[#dadce0] rounded-lg p-3 text-center" style={{ background: p.bg }}>
                    <p className="font-bold text-[#202124] text-lg">{p.phase}</p>
                    <p className="text-xs text-[#3c4043] mt-0.5">{p.name}</p>
                    <p className="text-[10px] font-mono text-[#5f6368] mt-1 border-t border-[#dadce0] pt-1">{p.review}</p>
                  </div>
                ))}
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">Major Reviews (NPR 7123.1C)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      {['Review','Full Name','Phase','Purpose'].map(h => (
                        <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['MCR','Mission Concept Review','Pre-A/A','Evaluates mission concepts and feasibility'],
                      ['SRR','System Requirements Review','A','Completeness and consistency of requirements'],
                      ['SDR','System Definition Review','A/B','System definition and allocated requirements'],
                      ['PDR','Preliminary Design Review','B','Preliminary design and verification approach'],
                      ['CDR','Critical Design Review','C','Detailed design readiness for fabrication'],
                      ['TRR','Test Readiness Review','C/D','Readiness for formal test execution'],
                      ['SIR','System Integration Review','D','System integration completeness'],
                      ['ORR','Operational Readiness Review','D','Readiness for operational deployment'],
                      ['FRR','Flight Readiness Review','E','Final go/no-go for each flight'],
                      ['DR','Decommissioning Review','E/F','Plans for safe disposal or decommissioning'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono font-bold text-[#0b3d91]">{row[0]}</td>
                        {row.slice(1).map((cell, j) => (
                          <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== 1000 SYSTEMS ENGINEERING ===== */}
            <section id="syseng" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">1000 Series — Systems Engineering &amp; Project Management</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Digital engineering, systems modeling, mission architecture, space system protection, and planetary environment testing. 5 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-HDBK-1004', title: 'NASA Digital Engineering Acquisition Framework Handbook', date: '2020-04', type: 'HDBK' },
                { code: 'NASA-HDBK-1005', title: 'NASA Space Mission Architecture Framework (SMAF) for Uncrewed Missions', date: '2021-03', type: 'HDBK' },
                { code: 'NASA-HDBK-1009A', title: 'NASA Systems Modeling Handbook for Systems Engineering', date: '2025-03', type: 'HDBK' },
                { code: 'NASA-STD-1006A', title: 'Space System Protection Standard', date: '2022-07', type: 'STD' },
                { code: 'NASA-STD-1008', title: 'Classifications & Requirements for Testing Systems Exposed to Dust in Planetary Environments', date: '2021-09', type: 'STD' },
              ]} />
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { label: 'Core SE', code: 'NPR 7123.1C', title: 'SE Processes & Requirements', desc: 'Defines the NASA "SE Engine" — 17 processes aligned with ISO/IEC 15288. Compliance matrix in Appendix H.' },
                  { label: 'PM', code: 'NPR 7120.5F', title: 'Space Flight PM Requirements', desc: 'Master PM framework for all space flight programs. Defines lifecycle, KDPs, reviews, WBS, and product-based approach.' },
                  { label: 'Digital', code: 'NASA-HDBK-1009A', title: 'Digital Engineering & MBSE', desc: 'NASA-HDBK-1004 (acquisition framework), HDBK-1009A (SysML/MBSE guidance, 2025). Digital Thread and Digital Twin practices.' },
                ].map(c => (
                  <div key={c.code} className="border border-[#dadce0] rounded-xl p-4 hover:border-[#0b3d91] transition-colors">
                    <span className="text-xs font-mono bg-[#e8eeff] text-[#0b3d91] px-2 py-0.5 rounded">{c.label}</span>
                    <h4 className="font-semibold text-[#202124] text-sm mt-2 mb-1">{c.title}</h4>
                    <p className="text-xs text-[#5f6368]">{c.desc}</p>
                  </div>
                ))}
              </div>
            </section>

            {/* ===== 2000 COMMS & IT ===== */}
            <section id="comms" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">2000 Series — Communications &amp; Information Technology</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Standards for digital television, audio/video, and imagery metadata. The smallest discipline category — 3 active standards.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-2818', title: 'Digital Television for NASA', date: 'Ver. 4.0 (2015)', type: 'STD' },
                { code: 'NASA-STD-2821', title: 'Audio and Video Standards for Internet Resources', date: 'V2 (2020)', type: 'STD' },
                { code: 'NASA-STD-2822', title: 'Still and Motion Imagery Metadata Standard', date: '2.0 (2024)', type: 'STD' },
              ]} />
              <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-lg p-4 text-sm text-[#3c4043]">
                Additional IT/communications requirements are addressed through <NprLink code="NPR 7120.7" /> (IT Program Management) and center-specific standards.
              </div>
            </section>

            {/* ===== 3000 HUMAN SYSTEMS ===== */}
            <section id="human" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">3000 Series — Human Systems &amp; Human-Rating</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Standards ensuring crew health, safety, and the human-rating certification process for crewed space systems.</p>
              <StdTable showDate rows={[
                { code: 'NPR 8705.2C', title: 'Human-Rating Requirements for Space Systems', date: 'Current', type: 'NPR' },
                { code: 'NASA-STD-3001 Vol.1', title: 'Spaceflight Human-System Standard — Crew Health (Vol 1C)', date: '2023', type: 'STD' },
                { code: 'NASA-STD-3001 Vol.2', title: 'Spaceflight Human-System Standard — Human Factors, Habitability (Vol 2E)', date: '2025', type: 'STD' },
                { code: 'NASA-STD-8719.29', title: 'Technical Requirements for Human-Rating', date: '—', type: 'STD' },
              ]} />
              <div className="bg-[#e8eeff] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e3a8a]">
                <strong>Human-Rating Certification:</strong> Required before first crewed mission. Four certification elements: Programmatic, Design, Production, Operations. Compliance verified at SRR, SDR, PDR, CDR, SIR, ORR. Director, JSC accepts crew risk.
              </div>
            </section>

            {/* ===== 4000 ELECTRICAL ===== */}
            <section id="electrical" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">4000 Series — Electrical &amp; Electronic Engineering</h2>
              <div className="h-1 w-12 bg-[#f59e0b] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Spacecraft electrical systems, charging, lightning protection, high-voltage design, and programmable logic devices. 9 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-4003A', title: 'Electrical Bonding for Launch Vehicles, Spacecraft, Payloads & Flight Equipment', date: '2013 (Chg 2016)', type: 'STD' },
                { code: 'NASA-STD-4005A', title: 'Low Earth Orbit Spacecraft Charging Design Standard', date: '2016 (Chg 2021)', type: 'STD' },
                { code: 'NASA-STD-4010A', title: 'Lightning Launch Commit Criteria for Space Flight', date: '2023', type: 'STD' },
                { code: 'NASA-HDBK-4001A', title: 'Electrical Grounding Architecture for Unmanned Spacecraft', date: '2025', type: 'HDBK' },
                { code: 'NASA-HDBK-4002B', title: 'Mitigating In-Space Charging Effects — A Guideline', date: '2022', type: 'HDBK' },
                { code: 'NASA-HDBK-4006A', title: 'Low Earth Orbit Spacecraft Charging Design Handbook', date: '2018', type: 'HDBK' },
                { code: 'NASA-HDBK-4007', title: 'Spacecraft High-Voltage Paschen and Corona Design Handbook', date: '2016 (Chg 2020)', type: 'HDBK' },
                { code: 'NASA-HDBK-4008', title: 'Programmable Logic Devices (PLD) Handbook', date: '2013 (Chg 2025)', type: 'HDBK' },
                { code: 'NASA-HDBK-4011', title: 'VHDL Style Handbook', date: '2022', type: 'HDBK' },
              ]} />
            </section>

            {/* ===== 5000 STRUCTURES ===== */}
            <section id="structures" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">5000 Series — Structures, Mechanical &amp; Propulsion</h2>
              <div className="h-1 w-12 bg-[#3b82f6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Structural factors of safety, fracture control, mechanisms, load analyses, fasteners, NDE, GSE design, and additive manufacturing. 13 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-5001B', title: 'Structural Design and Test Factors of Safety for Spaceflight Hardware', date: '2014 (Chg 2022)', type: 'STD' },
                { code: 'NASA-STD-5002A', title: 'Load Analyses of Spacecraft and Payloads', date: '2019', type: 'STD' },
                { code: 'NASA-STD-5005D', title: 'Design and Fabrication of Ground Support Equipment', date: '2013 (Chg 2024)', type: 'STD' },
                { code: 'NASA-STD-5009C', title: 'NDE Requirements for Fracture Critical Metallic Components', date: '2023', type: 'STD' },
                { code: 'NASA-STD-5012B', title: 'Strength & Life Assessment — Liquid-Fueled Propulsion System Engines', date: '2016', type: 'STD' },
                { code: 'NASA-STD-5017B', title: 'Design and Development Requirements for Mechanisms', date: '2022', type: 'STD' },
                { code: 'NASA-STD-5018', title: 'Strength Design for Glass, Ceramics & Windows in Human Spaceflight', date: '2011 (Chg 2017)', type: 'STD' },
                { code: 'NASA-STD-5019A', title: 'Fracture Control Requirements for Spaceflight Hardware', date: '2016 (Chg 2024)', type: 'STD' },
                { code: 'NASA-STD-5020B', title: 'Requirements for Threaded Fastening Systems in Spaceflight Hardware', date: '2021', type: 'STD' },
                { code: 'NASA-SPEC-5022', title: 'Manufacturing & Test Requirements for Pyrovalves', date: '2015 (Chg 2021)', type: 'SPEC' },
                { code: 'NASA-HDBK-5010', title: 'Fracture Control Implementation Handbook — Guidance (Vol 1 Rev A)', date: '2023', type: 'HDBK' },
                { code: 'NASA-HDBK-5010', title: 'Fracture Control Implementation Handbook — Examples (Vol 2 Rev A)', date: '2024', type: 'HDBK' },
                { code: 'NASA-HDBK-5026', title: 'Strength, Fatigue & Fracture Control for Additive Manufacturing Spaceflight Hardware', date: '2024', type: 'HDBK' },
              ]} />
            </section>

            {/* ===== 6000 MATERIALS ===== */}
            <section id="materials" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">6000 Series — Materials &amp; Processes</h2>
              <div className="h-1 w-12 bg-[#10b981] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Flammability, corrosion, materials selection, additive manufacturing, atomic oxygen durability. 7 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-6001B', title: 'Flammability, Offgassing & Compatibility Requirements and Test Procedures', date: '2011 (Chg 2025)', type: 'STD' },
                { code: 'NASA-STD-6012A', title: 'Corrosion Protection for Space Flight Hardware', date: '2022', type: 'STD' },
                { code: 'NASA-STD-6016C', title: 'Standard Materials & Processes Requirements for Spacecraft', date: '2021 (Chg 2023)', type: 'STD' },
                { code: 'NASA-STD-6030', title: 'Additive Manufacturing Requirements for Spaceflight Systems', date: '2021', type: 'STD' },
                { code: 'NASA-STD-6033', title: 'AM Requirements for Equipment and Facility Control', date: '2021', type: 'STD' },
                { code: 'NASA-HDBK-6007B', title: 'Material Removal Processes for Advanced Ceramic Components', date: '2018 (Chg 2022)', type: 'HDBK' },
                { code: 'NASA-HDBK-6024', title: 'Spacecraft Polymers Atomic Oxygen Durability Handbook', date: '2014 (Chg 2022)', type: 'HDBK' },
              ]} />
            </section>

            {/* ===== 7000 TEST & ENVIRONMENT ===== */}
            <section id="test-env" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">7000 Series — Test, Environment &amp; Modeling</h2>
              <div className="h-1 w-12 bg-[#8b5cf6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Vibroacoustic testing, payload requirements, models &amp; simulations, leak testing, and DFAT. 7 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-7001B', title: 'Payload Vibroacoustic Test Criteria', date: '2017', type: 'STD' },
                { code: 'NASA-STD-7002B', title: 'Payload Test Requirements', date: '2018 (Chg 2023)', type: 'STD' },
                { code: 'NASA-STD-7009B', title: 'Standard for Models and Simulations', date: '2024', type: 'STD' },
                { code: 'NASA-STD-7012A', title: 'Leak Test Requirements', date: '2023', type: 'STD' },
                { code: 'NASA-STD-1008', title: 'Dust Testing in Planetary Environments', date: '2021', type: 'STD' },
                { code: 'NASA-HDBK-7009A', title: 'Models & Simulations Implementation Guide', date: '2019', type: 'HDBK' },
                { code: 'NASA-HDBK-7010', title: 'Direct Field Acoustic Testing (DFAT)', date: '2016', type: 'HDBK' },
              ]} />
            </section>

            {/* ===== 8000 SAFETY ===== */}
            <section id="safety" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">8000 Series — Safety, Quality &amp; Mission Assurance</h2>
              <div className="h-1 w-12 bg-[#ef4444] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">The largest discipline category. Covers system safety, SMA management, workmanship, orbital debris, software assurance, and safety culture. ~35 documents.</p>

              <h3 className="font-semibold text-[#202124] mb-3">Key Safety &amp; SMA Standards</h3>
              <StdTable showDate rows={[
                { code: 'NASA-STD-8709.20', title: 'Management of SMA Technical Authority Requirements', date: '—', type: 'STD' },
                { code: 'NASA-HDBK-8709.22', title: 'SMA Acronyms, Abbreviations & Definitions', date: '2018 (Chg 2021)', type: 'HDBK' },
                { code: 'NASA-HDBK-8709.24', title: 'NASA Safety Culture Handbook', date: '2015 (Chg 2021)', type: 'HDBK' },
                { code: 'NASA-HDBK-8709.25', title: 'Human Factors Handbook — Procedural Guidance & Tools', date: '2023', type: 'HDBK' },
                { code: 'NASA-STD-8719.11B', title: 'Fire Protection and Life Safety', date: '2020', type: 'STD' },
                { code: 'NASA-STD-8719.12A', title: 'Safety Standard for Explosives, Propellants & Pyrotechnics', date: '2018', type: 'STD' },
                { code: 'NASA-STD-8719.17D', title: 'Orbital Debris Mitigation — NASA Requirements', date: '2023', type: 'STD' },
                { code: 'NASA-STD-8719.27', title: 'Implementing Planetary Protection Requirements', date: '2022', type: 'STD' },
                { code: 'NASA-HDBK-8715.26', title: 'Nuclear Flight Safety for Space Nuclear Systems', date: '2023', type: 'HDBK' },
              ]} />

              <h3 className="font-semibold text-[#202124] mb-3 mt-6">Workmanship Standards (8739 Series)</h3>
              <StdTable rows={[
                { code: 'NASA-STD-8739.1', title: 'Workmanship Standard for Polymeric Application on Electronic Assemblies', type: 'STD' },
                { code: 'NASA-STD-8739.4', title: 'Crimping, Interconnect Cables, Harnesses & Wiring', type: 'STD' },
                { code: 'NASA-STD-8739.5', title: 'Fiber Optic Terminations, Cable Assemblies & Installation', type: 'STD' },
                { code: 'NASA-STD-8739.6B', title: 'Implementation Requirements for NASA Approved Workmanship Standards', type: 'STD' },
                { code: 'NASA-STD-8739.8B', title: 'Software Assurance and Software Safety', type: 'STD' },
                { code: 'NASA-STD-8739.14', title: 'Fastener Procurement, Receiving Inspection & Storage for Mission Hardware', type: 'STD' },
                { code: 'NASA-HDBK-8739.18', title: 'Procedural Handbook for Problems, Nonconformances & Anomalies', type: 'HDBK' },
                { code: 'NASA-HDBK-8739.19', title: 'Measurement Quality Assurance Handbook (Annexes 2–4)', type: 'HDBK' },
                { code: 'NASA-HDBK-8739.21', title: 'Workmanship Manual for ESD Control', type: 'HDBK' },
              ]} />
            </section>

            {/* ===== 10000 FACILITIES ===== */}
            <section id="facilities" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">10000 Series — Facilities Design &amp; Infrastructure</h2>
              <div className="h-1 w-12 bg-[#6b7280] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Building Information Modeling (BIM) requirements and facility design standards for NASA centers. 2 documents.</p>
              <StdTable showDate rows={[
                { code: 'NASA-STD-10001', title: 'NASA Building Information Modeling Scope of Services & Requirements for Architects and Engineers', date: '2020', type: 'STD' },
                { code: 'NASA-STD-10002', title: 'NASA Facilities Design Standard', date: '2021', type: 'STD' },
              ]} />
              <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-lg p-4 text-sm text-[#3c4043]">
                NASA-STD-10001 mandates BIM for all new construction and major renovations. Additional facility project requirements are managed under NPR 8820.2 (Facility Project Requirements).
              </div>
            </section>

            {/* ===== SOFTWARE ===== */}
            <section id="software" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Software Engineering <span className="text-sm font-normal text-[#5f6368]">(Cross-Cutting)</span></h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <StdTable rows={[
                { code: 'NPR 7150.2C', title: 'NASA Software Engineering Requirements — defines classification A–E and lifecycle requirements', type: 'NPR' },
                { code: 'NASA-STD-8739.8B', title: 'Software Assurance and Software Safety Standard', type: 'STD' },
                { code: 'NASA-HDBK-2203', title: 'NASA Software Engineering Handbook (Rev B, wiki-based)', type: 'HDBK' },
                { code: 'NASA-STD-7009B', title: 'Standard for Models and Simulations — M&S credibility assessment', type: 'STD' },
              ]} />
              <div className="bg-[#f8f9fa] border border-[#dadce0] rounded-xl p-5">
                <h3 className="font-semibold text-[#202124] mb-3">Software Classification (NPR 7150.2C)</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className="bg-white text-left">
                        {['Class','Criticality','Example'].map(h => <th key={h} className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ['A','Catastrophic failure impact','Human-rated flight control'],
                        ['B','Critical failure impact','Mission critical systems'],
                        ['C','Significant failure impact','Science payload software'],
                        ['D','Negligible failure impact','Ground tools'],
                        ['E','Information only (legacy/COTS)','Off-the-shelf OS'],
                      ].map((row, i) => (
                        <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] font-bold text-[#0b3d91]">{row[0]}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#3c4043]">{row[1]}</td>
                          <td className="px-3 py-2 border-b border-[#f1f3f4] text-[#5f6368]">{row[2]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>

            {/* ===== COMMERCIAL OVERVIEW ===== */}
            <section id="commercial" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Commercial Space Programs — Overview</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">NASA&apos;s public-private partnership model — from ISS cargo in 2006 to lunar landing services and post-ISS commercial stations.</p>

              <h3 className="font-semibold text-[#202124] mb-3">Program Timeline (2006–2025)</h3>
              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-[#f8f9fa] text-left">
                      {['Year','Program','Milestone'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ['2006','COTS','Commercial Orbital Transportation Services initiated'],
                      ['2010','CCDev1','$50M SAAs — crew transport concept development (5 companies)'],
                      ['2011','CCDev2','$269M for crew vehicle development (Blue Origin, Boeing, SNC, SpaceX)'],
                      ['2012','CRS-1','First Commercial Resupply contracts (SpaceX Dragon, Orbital Cygnus)'],
                      ['2012','CCiCap','$1.1B — end-to-end crew systems (Boeing, SNC, SpaceX)'],
                      ['2014','CCtCap','$6.8B FAR contracts — Boeing ($4.82B), SpaceX ($3.14B)'],
                      ['2016','CRS-2','2nd-gen resupply (SpaceX, Northrop, Sierra Nevada / Dream Chaser)'],
                      ['2018','CLPS','Commercial Lunar Payload Services — $2.6B IDIQ pool initiated'],
                      ['2020','CCP Ops','SpaceX Crew Dragon Demo-2 — first operational commercial crew'],
                      ['2020','HLS','SpaceX Starship HLS selected ($2.9B) for Artemis crewed landings'],
                      ['2021','CLD','$415.6M Phase 1 awards — Axiom, Starlab, Orbital Reef'],
                      ['2023','HLS Sustain','Blue Origin Blue Moon selected as sustaining HLS ($3.4B)'],
                      ['2024','CLPS Landings','IM-1 Odysseus (success), Blue Ghost-1 (success), Peregrine-1 (failed)'],
                      ['2025','CLD Phase 2','Revised AFP — $2.1B over 5 years, min 2 providers'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono font-bold text-[#0b3d91] text-xs">{row[0]}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-semibold text-[#202124] text-xs">{row[1]}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{row[2]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-[#f0fdf4] border border-[#86efac] rounded-lg p-4 text-sm text-[#14532d]">
                <strong>Partnership Model:</strong> SAAs for development (milestone-based, company retains IP and design authority) → CPC/CCtCap for certification → FAR firm-fixed-price for operations. Commercial providers own the system and can sell capacity to other customers. Risk tolerance is higher; costs are 4–10× lower vs. traditional acquisition.
              </div>
            </section>

            {/* ===== COTS & CRS ===== */}
            <section id="cots-crs" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">COTS &amp; Commercial Resupply Services (CRS)</h2>
              <div className="h-1 w-12 bg-[#10b981] rounded mb-6"></div>

              <h3 className="font-semibold text-[#202124] mb-3">COTS — Commercial Orbital Transportation Services (2006–2013)</h3>
              <div className="overflow-x-auto mb-4">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Company','Vehicle','Award','Result'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['SpaceX','Dragon + Falcon 9','$396M','First private spacecraft to berth ISS (May 2012)'],
                      ['Orbital Sciences (now Northrop)','Cygnus + Antares','$288M','First Cygnus ISS berthing (Jan 2014)'],
                      ['Rocketplane Kistler','K-1 (terminated)','$207M rescinded','Contract terminated Oct 2007 — non-performance'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        {row.map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">CRS-1 (2012–2024)</h3>
              <div className="overflow-x-auto mb-4">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Company','Vehicle','Contract','Missions'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[['SpaceX','Dragon (Cargo)','$1.6B','20 missions (CRS-1 through CRS-20)'],['Northrop Grumman','Cygnus','$1.9B','11 missions (OA/NG-1 through NG-11)']].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        {row.map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">CRS-2 (2016–present)</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Company','Vehicle','Contract','Key Features'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['SpaceX','Cargo Dragon 2','~$3.04B','Autonomous docking, pressurized + unpressurized cargo'],
                      ['Northrop Grumman','Cygnus (Enhanced)','~$3.04B','Disposal capability, extended missions'],
                      ['Sierra Space','Dream Chaser','~$3.04B','Winged vehicle, runway landing, cargo return, first flight 2025'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        {row.map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== CCP ===== */}
            <section id="ccp" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Commercial Crew Program (CCP)</h2>
              <div className="h-1 w-12 bg-[#8b5cf6] rounded mb-6"></div>

              <h3 className="font-semibold text-[#202124] mb-3">Program Phases &amp; Funding ($8.2B+ Total)</h3>
              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Phase','Mechanism','Period','Funding','Companies'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['CCDev1','Space Act Agreement','2010','$50M','Blue Origin ($3.7M), Boeing ($18M), SNC ($20M), ULA ($6.7M)'],
                      ['CCDev2','Space Act Agreement','2011–14','$269M','Blue Origin ($22M), Boeing ($92M), SNC ($80M), SpaceX ($75M)'],
                      ['CCiCap','Space Act Agreement','2012–14','$1.1B','Boeing ($460M), SNC ($212M), SpaceX ($440M)'],
                      ['CPC','FAR Contract','2012–14','$30M','Boeing, SNC, SpaceX (~$10M each)'],
                      ['CCtCap','FAR FFP','2014–present','$6.8B','Boeing ($4.82B), SpaceX ($3.14B)'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono font-bold text-[#0b3d91] text-xs">{row[0]}</td>
                        {row.slice(1).map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-[#3c4043] text-xs">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-6">
                <div className="border border-[#dadce0] rounded-xl p-4" style={{ borderTopColor: '#06b6d4', borderTopWidth: 3 }}>
                  <span className="text-xs font-mono bg-[#e0f2fe] text-[#0369a1] px-2 py-0.5 rounded">Operational</span>
                  <h3 className="font-semibold text-[#202124] mt-2 mb-1">SpaceX Crew Dragon</h3>
                  <p className="text-xs text-[#5f6368]">Capsule, 4 crew + cargo. Autonomous docking. Falcon 9 launch. First operational flight Crew-1 (Nov 2020). 10+ crew rotation missions through 2025. Also: Ax-1/2/3, Inspiration4, Polaris Dawn.</p>
                  <p className="text-xs font-mono text-[#0b3d91] mt-2">CCtCap: $3.14B · Operational since 2020</p>
                </div>
                <div className="border border-[#dadce0] rounded-xl p-4" style={{ borderTopColor: '#f59e0b', borderTopWidth: 3 }}>
                  <span className="text-xs font-mono bg-[#fef3c7] text-[#92400e] px-2 py-0.5 rounded">Limited Ops</span>
                  <h3 className="font-semibold text-[#202124] mt-2 mb-1">Boeing CST-100 Starliner</h3>
                  <p className="text-xs text-[#5f6368]">OFT-2 (May 2022) success. CFT (Jun 2024) docked but returned uncrewed due to thruster issues. Crew returned on SpaceX Crew-9. Boeing reviewing future of space business.</p>
                  <p className="text-xs font-mono text-[#0b3d91] mt-2">CCtCap: $4.82B · Certification: In review</p>
                </div>
              </div>

              <div className="bg-[#fff1f2] border border-[#fecdd3] rounded-lg p-4 text-sm text-[#9f1239]">
                <strong>CCT-STD-1140</strong> — Crew Transportation Technical Standards — master requirements document. Covers all mission phases: pre-launch, ascent, orbital, rendezvous/docking, ISS-attached, undocking, re-entry, landing. Providers may propose Alternate Standards and request Variances.
              </div>
            </section>

            {/* ===== CLPS ===== */}
            <section id="clps" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Commercial Lunar Payload Services (CLPS)</h2>
              <div className="h-1 w-12 bg-[#3b82f6] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-3">Rapid, affordable robotic lunar payload delivery for Artemis pathfinding. IDIQ pool — $2.6B max through Nov 2028. 14 eligible U.S. vendors. NASA buys service, not hardware.</p>

              <div className="grid md:grid-cols-2 gap-4 mb-6">
                {[
                  { name: 'Intuitive Machines', detail: 'Nova-C lander. Houston, TX. 4 task orders. IM-1 (Feb 2024 — first commercial lunar landing), IM-2 (2025, south pole), IM-3 (2025–26, Reiner Gamma), IM-4 (2027).', color: '#10b981' },
                  { name: 'Firefly Aerospace', detail: 'Blue Ghost lander. Cedar Park, TX. 4 task orders. BG-1 (Jan 2025, Mare Crisium — success), BG-2 (2026, far side), BG-3 (2027), BG-4 (2028, CSA rover).', color: '#f59e0b' },
                  { name: 'Astrobotic Technology', detail: 'Peregrine & Griffin landers. Pittsburgh, PA. 2 task orders. Peregrine-1 (Jan 2024 — propellant leak, failed). Griffin (2025 — south pole).', color: '#3b82f6' },
                  { name: 'Draper / Team Draper', detail: 'APEX 1.0 lander. Cambridge, MA. 1 task order. Schrödinger Basin far side (2026) — seismometers, heat flow, subsurface measurements.', color: '#8b5cf6' },
                  { name: 'Blue Origin', detail: 'Blue Moon Mark 1. Kent, WA. 1 task order. Demo flight (2025). Also delivering VIPER rover (2027).', color: '#0b3d91' },
                  { name: 'Other Eligible Vendors', detail: 'Ceres Robotics, Deep Space Systems, Lockheed Martin, Masten (bankrupt — assets to Astrobotic), Moon Express, Sierra Space, SpaceX, Tyvak.', color: '#6b7280' },
                ].map(p => (
                  <div key={p.name} className="border border-[#dadce0] rounded-xl p-4" style={{ borderTopColor: p.color, borderTopWidth: 3 }}>
                    <h4 className="font-semibold text-[#202124] text-sm mb-1">{p.name}</h4>
                    <p className="text-xs text-[#5f6368]">{p.detail}</p>
                  </div>
                ))}
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">Mission Status</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Mission','Provider','Date','Site','Status','Value'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      { m: 'Peregrine-1', p: 'Astrobotic', d: 'Jan 2024', s: 'Lacus Mortis (planned)', st: '✗ Failed — propellant leak', stColor: '#dc2626', v: '$108M' },
                      { m: 'IM-1 (Odysseus)', p: 'Intuitive Machines', d: 'Feb 2024', s: 'Malapert A (south pole)', st: '✓ Landed (tipped)', stColor: '#16a34a', v: '$118M' },
                      { m: 'Blue Ghost-1', p: 'Firefly', d: 'Jan 2025', s: 'Mare Crisium', st: '✓ Landed — success', stColor: '#16a34a', v: '$93.3M' },
                      { m: 'IM-2', p: 'Intuitive Machines', d: '2025', s: 'Mons Mouton (south pole)', st: '⚠ Landed (tipped)', stColor: '#d97706', v: '$47M' },
                      { m: 'Griffin', p: 'Astrobotic', d: '2025', s: 'Mons Mouton', st: 'In preparation', stColor: '#6b7280', v: '$320.4M' },
                      { m: 'IM-3', p: 'Intuitive Machines', d: '2025–26', s: 'Reiner Gamma', st: 'Manifested', stColor: '#6b7280', v: '$77.5M' },
                      { m: 'Blue Moon Demo', p: 'Blue Origin', d: '2025', s: 'TBD', st: 'In development', stColor: '#6b7280', v: '—' },
                      { m: 'Draper SERIES-2', p: 'Team Draper', d: '2026', s: 'Schrödinger (far side)', st: 'In development', stColor: '#6b7280', v: '$73M' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono text-xs text-[#0b3d91] font-semibold">{row.m}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{row.p}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.d}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.s}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs font-medium" style={{ color: row.stColor }}>{row.st}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== CLD ===== */}
            <section id="cld" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Commercial LEO Destinations (CLD)</h2>
              <div className="h-1 w-12 bg-[#f59e0b] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-3">Privately-owned stations to replace ISS (~2030 retirement). <strong>Critical timeline:</strong> NASA&apos;s OIG warns commercial platform unlikely before 2030. FY2026: $272.3M requested; $2.1B projected over 5 years.</p>

              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Phase','Mechanism','Scope','Status'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['Phase 1 — CDISS','FFP Contract ($140M)','Axiom Space — ISS-attached modules → free-flying','Active (AxH1 targeted 2026–27)'],
                      ['Phase 1 — CDFF','Funded SAAs ($415.6M)','Free-flying station design maturation through 2025','Complete'],
                      ['Phase 2','Funded SAAs (revised 2025)','CDR readiness + in-space crewed demo. Min 2 providers. ~$1.5B','AFP expected 2025, awards 2026'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        {row.map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="grid md:grid-cols-2 gap-4 mb-6">
                {[
                  { name: 'Axiom Station', badge: 'CDISS', color: '#f97316', desc: 'Modules attached to ISS, then detach for free-flying ops. AxH1 module targeted 2026–27. Has flown Ax-1/2/3 private missions. $140M NASA + private investment.' },
                  { name: 'Starlab', badge: 'CDFF — $160M', color: '#8b5cf6', desc: 'Voyager Space + Airbus + Northrop. Single-launch on Starship. 4 crew. George Washington Carver Science Park labs. Power/volume ≈ ISS. Target: 2028–29.' },
                  { name: 'Orbital Reef', badge: 'CDFF — $130M', color: '#0b3d91', desc: 'Blue Origin + Sierra Space + Boeing. Modular "business park". Up to 10 crew, 830 m³. Includes LIFE inflatable habitat. Progress slowed 2024.' },
                  { name: 'Vast — Haven-1/2', badge: 'Unfunded SAA', color: '#10b981', desc: 'Haven-1: single module, Q1 2027 on Falcon 9, 4 crew for 2-week missions. Haven-2: larger multi-module. Bidding for CLD Phase 2. Private funding from Jed McCaleb.' },
                ].map(p => (
                  <div key={p.name} className="border border-[#dadce0] rounded-xl p-4" style={{ borderTopColor: p.color, borderTopWidth: 3 }}>
                    <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: p.color + '20', color: p.color }}>{p.badge}</span>
                    <h4 className="font-semibold text-[#202124] text-sm mt-2 mb-1">{p.name}</h4>
                    <p className="text-xs text-[#5f6368]">{p.desc}</p>
                  </div>
                ))}
              </div>

              <div className="bg-[#fffbeb] border border-[#fde68a] rounded-lg p-4 text-sm text-[#78350f]">
                <strong>2025 Phase 2 Revised Strategy:</strong> Goal adjusted to 4-person crews for month-long missions. SAA includes milestones to CDR readiness + in-space crewed demo (non-NASA crew). Min 25% paid after successful demo. Min 2, preferably 3+ providers. International partners welcome.
              </div>
            </section>

            {/* ===== HLS ===== */}
            <section id="hls" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Human Landing System (HLS)</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Commercial lunar landers enabling Artemis crewed surface missions — the first human Moon landings since Apollo 17 in 1972.</p>
              <div className="grid md:grid-cols-2 gap-5 mb-6">
                <div className="border border-[#dadce0] rounded-xl p-5" style={{ borderTopColor: '#06b6d4', borderTopWidth: 3 }}>
                  <span className="text-xs font-mono bg-[#e0f2fe] text-[#0369a1] px-2 py-0.5 rounded">Initial Provider</span>
                  <h3 className="font-semibold text-[#202124] mt-2 mb-2">SpaceX Starship HLS</h3>
                  <p className="text-xs text-[#5f6368]">Contract: $2.89B (Apr 2021), augmented for Artemis IV. Based on Starship architecture with orbital refueling. Will land 2 astronauts on lunar south pole for Artemis III. Requires multiple tanker launches. First uncrewed demo required before crewed mission.</p>
                  <p className="text-xs font-mono text-[#0b3d91] mt-2">Artemis III · Artemis IV</p>
                </div>
                <div className="border border-[#dadce0] rounded-xl p-5" style={{ borderTopColor: '#f59e0b', borderTopWidth: 3 }}>
                  <span className="text-xs font-mono bg-[#fef3c7] text-[#92400e] px-2 py-0.5 rounded">Sustaining Provider</span>
                  <h3 className="font-semibold text-[#202124] mt-2 mb-2">Blue Origin Blue Moon</h3>
                  <p className="text-xs text-[#5f6368]">Contract: $3.4B (May 2023). Mark 2 lander for Artemis V and beyond. Partners: Lockheed Martin, Draper, Boeing, Astrobotic, Honeybee Robotics. Designed for reusability. Mark 1 demo under separate CLPS contract.</p>
                  <p className="text-xs font-mono text-[#0b3d91] mt-2">Artemis V+ · Blue Moon Mark 1 & 2</p>
                </div>
              </div>
              <div className="bg-[#e8eeff] border border-[#93c5fd] rounded-lg p-4 text-sm text-[#1e3a8a]">
                <strong>HLS Certification:</strong> CCtCap-like framework adapted for lunar missions. Providers comply with <NprLink code="NPR 8705.2C" /> intent, <NasaDocLink code="NASA-STD-3001 Vol.1" />, and safety requirements via Alternate Standards, VCNs, and Hazard Reports. NASA maintains Technical Authority oversight while providers retain design authority.
              </div>
            </section>

            {/* ===== ARTEMIS ===== */}
            <section id="artemis" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Artemis Program Architecture</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>

              <h3 className="font-semibold text-[#202124] mb-3">Mission Manifest</h3>
              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Mission','Date (NET)','Type','Crew','SLS','Key Objectives','Status'].map(h => <th key={h} className="px-3 py-2 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      { m:'Artemis I', d:'Nov 2022', t:'Uncrewed test', c:'—', sls:'Block 1', obj:'SLS maiden flight, Orion heat shield test, DRO 25.5-day mission', st:'✓ Complete', stC:'#16a34a' },
                      { m:'Artemis II', d:'Mar 2026', t:'Crewed flyby', c:'4', sls:'Block 1', obj:'First crewed SLS/Orion; lunar free-return ~10 days; crew: Wiseman, Glover, Koch, Hansen (CSA)', st:'On pad — wet dress rehearsal', stC:'#d97706' },
                      { m:'Artemis III', d:'NET 2028', t:'Crewed landing', c:'4 (2 land)', sls:'Block 1', obj:'First crewed lunar landing since 1972; south pole; Starship HLS; Axiom xEVAS suits. May be replanned.', st:'In development — delays', stC:'#6b7280' },
                      { m:'Artemis IV', d:'NET Sep 2028', t:'Gateway + Landing', c:'4', sls:'Block 1B', obj:'Deliver I-Hab to Gateway; first crewed Gateway visit; lunar surface via HLS', st:'In development', stC:'#6b7280' },
                      { m:'Artemis V', d:'NET 2030', t:'Gateway + Landing', c:'4', sls:'Block 1B', obj:'Deliver Lunar View; Blue Origin Blue Moon HLS; sustained operations', st:'Planning', stC:'#6b7280' },
                      { m:'Artemis VI–VIII', d:'2030s', t:'Sustained', c:'4', sls:'Block 1B/2', obj:'Routine Gateway ops; extended surface stays; ISRU demos; Mars prep', st:'Planning', stC:'#6b7280' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] font-mono text-xs font-bold text-[#0b3d91]">{row.m}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.d}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{row.t}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.c}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs font-mono text-[#5f6368]">{row.sls}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{row.obj}</td>
                        <td className="px-3 py-2 border-b border-[#f1f3f4] text-xs font-medium" style={{ color: row.stC }}>{row.st}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="bg-[#fffbeb] border border-[#fde68a] rounded-lg p-3 text-xs text-[#78350f] mb-6">
                <strong>⚠ Schedule Uncertainty (Feb 2026):</strong> Artemis II on pad at LC-39B targeting March 2026. Artemis III pushed to NET 2028 due to Starship HLS delays and Orion heat shield issues. FY2026 budget proposes cancelling SLS/Orion after Artemis III and the Gateway program, shifting to commercial alternatives. Funded through FY2032 by the One Big Beautiful Bill Act ($2.6B for Gateway).
              </div>

              <h3 className="font-semibold text-[#202124] mb-3">Architecture Elements</h3>
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                {[
                  { name: 'Space Launch System (SLS)', color: '#f97316', badge: 'Launch', desc: 'Block 1: 95 mT LEO, 27 mT TLI. 322 ft, 8.8M lbs thrust. 4× RS-25 + 2× 5-segment SRBs. ICPS upper stage. Block 1B: EUS for Artemis IV+, 38 mT TLI. Block 2 (future): 130 mT LEO. ~$4.1B per launch. MSFC managed, Boeing prime.', codes: ['NASA-STD-5001B', 'NASA-STD-5019A', 'NASA-STD-5020B'] },
                  { name: 'Orion MPCV', color: '#f59e0b', badge: 'Crew', desc: 'Deep space capsule, 4 crew, up to 30+ days with Gateway. LAS for full ascent abort capability. European Service Module (ESA/Airbus): 33 kN OMS-E, 4× solar arrays 11.1 kW. AVCOAT ablative heat shield. Skip-entry at ~25,000 mph / 5,000°F. JSC managed, Lockheed Martin prime.', codes: ['NPR 8705.2C', 'NASA-STD-3001 Vol.1'] },
                  { name: 'Gateway — Initial Config', color: '#8b5cf6', badge: 'Station', desc: 'PPE (Maxar, $375M): 60 kW SEP, 50 kW power, xenon ion AEPS thrusters. HALO (Northrop/Thales Alenia, $935M): first habitable module, C&DH, life support, docking ports. Launched together on Falcon Heavy (NET 2027). NRHO ~1,000–43,500 mi from Moon surface.', codes: [] },
                  { name: 'Gateway — Expansion', color: '#6366f1', badge: 'Future', desc: 'Lunar I-Hab (ESA/JAXA/Thales Alenia): enhanced habitation, ECLSS. Arrives Artemis IV. Lunar View (ESA): refueling, logistics. Canadarm3 (CSA): autonomous robotic arm. Crew & Science Airlock (MBRSC/UAE). International partners: ESA, JAXA, CSA, MBRSC.', codes: [] },
                ].map(item => (
                  <div key={item.name} className="border border-[#dadce0] rounded-xl p-4" style={{ borderTopColor: item.color, borderTopWidth: 3 }}>
                    <span className="text-xs font-mono px-2 py-0.5 rounded" style={{ background: item.color + '20', color: item.color }}>{item.badge}</span>
                    <h4 className="font-semibold text-[#202124] text-sm mt-2 mb-1">{item.name}</h4>
                    <p className="text-xs text-[#5f6368] mb-2">{item.desc}</p>
                    {item.codes.length > 0 && (
                      <div className="flex flex-wrap gap-1">
                        {item.codes.map(c => c.startsWith('NPR') ? <NprLink key={c} code={c} /> : <NasaDocLink key={c} code={c} />)}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <h3 className="font-semibold text-[#202124] mb-3 mt-6">Applicable Standards Across Artemis</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Domain','Standards','Application'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      { d:'Human Rating', codes:['NPR 8705.2C'], app:'SLS, Orion, HLS, xEVAS — certification before crewed missions' },
                      { d:'Crew Health', codes:['NASA-STD-3001 Vol.1'], app:'Crew health, human factors, habitability for all crewed elements' },
                      { d:'Structures', codes:['NASA-STD-5001B','NASA-STD-5019A'], app:'Factors of safety, fracture control for SLS, Orion, Gateway' },
                      { d:'Materials', codes:['NASA-STD-6016C','NASA-STD-6001B'], app:'M&P requirements, flammability/offgassing for pressurized volumes' },
                      { d:'Software', codes:['NPR 7150.2C','NASA-STD-8739.8B'], app:'Flight software — Orion, SLS avionics, Gateway C&DH' },
                      { d:'Safety', codes:['NPR 8715.3C','NASA-STD-8719.17D'], app:'General safety; orbital debris for Gateway' },
                      { d:'Test', codes:['NASA-STD-7001B','NASA-STD-7002B'], app:'Vibroacoustic, payload testing for all flight hardware' },
                      { d:'Systems Eng.', codes:['NPR 7123.1C','NPR 7120.5F'], app:'SE processes, lifecycle management across Artemis elements' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-medium text-[#202124] text-xs">{row.d}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4]">
                          <div className="flex flex-wrap gap-1">
                            {row.codes.map(c => c.startsWith('NPR') ? <NprLink key={c} code={c} /> : <NasaDocLink key={c} code={c} />)}
                          </div>
                        </td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.app}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== COMPLETE LIST ===== */}
            <section id="complete-list" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Complete Standards List</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">All 83 active NASA Technical Standards at <a href="https://standards.nasa.gov/all-standards" target="_blank" rel="noopener noreferrer" className="text-[#0b3d91] hover:underline font-medium">standards.nasa.gov/all-standards ↗</a></p>
              <div className="overflow-x-auto mb-6">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Series','Discipline','Count','Documents (STD/HDBK)'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['1000','Systems Engineering & Project Mgmt','5','3 HDBK, 2 STD'],
                      ['2000','Communications & IT','4','1 HDBK, 3 STD'],
                      ['3000','Human Systems','2','2 STD'],
                      ['4000','Electrical & Electronic','9','3 STD, 6 HDBK'],
                      ['5000','Structures, Mechanical, Propulsion','13','9 STD, 3 HDBK, 1 SPEC'],
                      ['6000','Materials & Processes','7','5 STD, 2 HDBK'],
                      ['7000','Test, Environment & Modeling','7','4 STD, 3 HDBK'],
                      ['8000','Safety, Quality, Reliability','~35','Largest category'],
                      ['10000','Facilities Design','2','2 STD'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono font-bold text-[#0b3d91] text-xs">{row[0]}</td>
                        {row.slice(1).map((cell, j) => <td key={j} className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{cell}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== NPR INDEX ===== */}
            <section id="npr-index" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Key NPR / NPD Index</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <p className="text-[#3c4043] mb-4">Available at <a href="https://nodis3.gsfc.nasa.gov" target="_blank" rel="noopener noreferrer" className="text-[#0b3d91] hover:underline">nodis3.gsfc.nasa.gov ↗</a></p>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Document','Title','Domain'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      { code:'NPD 1000.0', title:'NASA Governance and Strategic Management Handbook', domain:'Governance' },
                      { code:'NPD 7120.4', title:'NASA Engineering and Program/Project Management Policy', domain:'Policy' },
                      { code:'NPR 7120.5F', title:'NASA Space Flight Program and Project Management Requirements', domain:'Program Mgmt' },
                      { code:'NPR 7120.7', title:'NASA IT and Institutional Infrastructure PM Requirements', domain:'IT programs' },
                      { code:'NPR 7120.8', title:'NASA Research and Technology PM Requirements', domain:'R&T programs' },
                      { code:'NPR 7120.10B', title:'Technical Standards for NASA Programs and Projects', domain:'Standards gov.' },
                      { code:'NPR 7123.1C', title:'NASA Systems Engineering Processes and Requirements', domain:'Sys. Engineering' },
                      { code:'NPR 7150.2C', title:'NASA Software Engineering Requirements', domain:'Software' },
                      { code:'NPR 8000.4', title:'Agency Risk Management Procedural Requirements', domain:'Risk Mgmt' },
                      { code:'NPR 8705.2C', title:'Human-Rating Requirements for Space Systems', domain:'Human Rating' },
                      { code:'NPR 8705.4', title:'Risk Classification for NASA Payloads', domain:'Risk class.' },
                      { code:'NPR 8705.5', title:'Technical PRA Procedures for Safety and Mission Success', domain:'PRA' },
                      { code:'NPR 8705.6', title:'SMA Audits, Reviews, and Assessments', domain:'SMA oversight' },
                      { code:'NPR 8715.3C', title:'NASA General Safety Program Requirements', domain:'Safety' },
                      { code:'NPR 8735.2', title:'Management of QA Functions for NASA Contracts', domain:'Quality' },
                      { code:'NPR 8900.1', title:'Health and Medical Requirements for Human Space Exploration', domain:'Crew health' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4]"><NprLink code={row.code} /></td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#3c4043]">{row.title}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.domain}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            {/* ===== CROSS-REFERENCE ===== */}
            <section id="crossref" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Cross-Reference: NASA ↔ ECSS ↔ Industry</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Domain','NASA','ECSS','Industry Standards'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      { d:'Systems Engineering', nasa:'NPR 7123.1C', ecss:'ECSS-E-ST-10C Rev.1', ind:'ISO/IEC 15288, INCOSE SE HB' },
                      { d:'Program Management', nasa:'NPR 7120.5F', ecss:'ECSS-M-ST-10C Rev.1', ind:'PMI PMBOK' },
                      { d:'Human Rating', nasa:'NPR 8705.2C', ecss:'—', ind:'FAA 14 CFR Part 460' },
                      { d:'Software Engineering', nasa:'NPR 7150.2C', ecss:'ECSS-E-ST-40C Rev.1', ind:'DO-178C, IEEE 12207' },
                      { d:'Software Assurance', nasa:'NASA-STD-8739.8B', ecss:'ECSS-Q-ST-80C Rev.1', ind:'DO-178C, IEC 61508-3' },
                      { d:'Structural Design', nasa:'NASA-STD-5001B', ecss:'ECSS-E-ST-32C Rev.1', ind:'AIAA S-110, S-111' },
                      { d:'Fracture Control', nasa:'NASA-STD-5019A', ecss:'ECSS-E-ST-32-01C Rev.2', ind:'MIL-STD-1530' },
                      { d:'Safety', nasa:'NPR 8715.3C', ecss:'ECSS-Q-ST-40C Rev.1', ind:'MIL-STD-882E, IEC 61508' },
                      { d:'FMEA/FMECA', nasa:'NPR 8715.3 (ref)', ecss:'ECSS-Q-ST-30-02C', ind:'MIL-STD-1629, IEC 60812' },
                      { d:'Materials & Processes', nasa:'NASA-STD-6016C', ecss:'ECSS-Q-ST-70C Rev.2', ind:'ASTM, MIL-HDBK-5' },
                      { d:'Electrical Bonding', nasa:'NASA-STD-4003A', ecss:'ECSS-E-ST-20C Rev.2', ind:'MIL-STD-464' },
                      { d:'Orbital Debris', nasa:'NASA-STD-8719.17D', ecss:'ECSS-U-AS-10C Rev.2', ind:'ISO 24113' },
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-medium text-[#202124] text-xs">{row.d}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4]">
                          {row.nasa === '—' ? <span className="text-[#5f6368] text-xs">—</span> :
                            row.nasa.startsWith('NPR') ? <NprLink code={row.nasa} /> : <NasaDocLink code={row.nasa} />}
                        </td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono text-xs text-[#3c4043]">{row.ecss}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row.ind}</td>
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

            {/* ===== GLOSSARY ===== */}
            <section id="glossary" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Glossary &amp; Abbreviations</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-2">
                {[
                  ['CCDev','Commercial Crew Development'],['CCtCap','Commercial Crew Transportation Capability'],
                  ['CDR','Critical Design Review'],['CLD','Commercial LEO Destinations'],
                  ['CLPS','Commercial Lunar Payload Services'],['CoFR','Certification of Flight Readiness'],
                  ['COTS','Commercial Orbital Transportation Services'],['CPC','Certification Products Contract'],
                  ['CRS','Commercial Resupply Services'],['ETA','Engineering Technical Authority'],
                  ['EVA','Extravehicular Activity'],['FRR','Flight Readiness Review'],
                  ['Gateway','Lunar Orbital Platform-Gateway'],['GNC','Guidance, Navigation & Control'],
                  ['HALO','Habitation and Logistics Outpost'],['HLS','Human Landing System'],
                  ['HMTA','Health & Medical Technical Authority'],['HRCP','Human Rating Certification Package'],
                  ['ISS','International Space Station'],['IV&V','Independent Verification & Validation'],
                  ['JSC','Johnson Space Center'],['KDP','Key Decision Point'],
                  ['KSC','Kennedy Space Center'],['MBSE','Model-Based Systems Engineering'],
                  ['MSFC','Marshall Space Flight Center'],['NODIS','NASA Online Directives Information System'],
                  ['NPD','NASA Policy Directive'],['NPR','NASA Procedural Requirements'],
                  ['NRHO','Near-Rectilinear Halo Orbit'],['NTSP','NASA Technical Standards Program'],
                  ['OCE','Office of the Chief Engineer'],['OSMA','Office of Safety & Mission Assurance'],
                  ['PDR','Preliminary Design Review'],['PPE','Power and Propulsion Element'],
                  ['PRA','Probabilistic Risk Assessment'],['SAA','Space Act Agreement'],
                  ['SEMP','Systems Engineering Management Plan'],['SLS','Space Launch System'],
                  ['SMA','Safety & Mission Assurance'],['SRR','System Requirements Review'],
                  ['TA','Technical Authority'],['TLI','Trans-Lunar Injection'],
                  ['VCN','Verification Closure Notice'],['xEVAS','Axiom Extravehicular Mobility Unit'],
                ].map(([term, def]) => (
                  <div key={term} className="bg-[#f8f9fa] border border-[#dadce0] rounded-lg px-3 py-2.5">
                    <dt className="font-mono font-bold text-[#0b3d91] text-sm">{term}</dt>
                    <dd className="text-xs text-[#3c4043] mt-0.5">{def}</dd>
                  </div>
                ))}
              </div>
            </section>

            {/* ===== RESOURCES ===== */}
            <section id="resources" className="mb-16 scroll-mt-20">
              <h2 className="text-2xl font-bold text-[#202124] mb-2">Resources &amp; NASA Centers</h2>
              <div className="h-1 w-12 bg-[#0b3d91] rounded mb-6"></div>
              <div className="grid md:grid-cols-2 gap-4 mb-10">
                {[
                  { name:'NASA Technical Standards System (NTSS)', url:'https://standards.nasa.gov', desc:'All NASA-STD, HDBK, SPEC documents + OCE Endorsed Standards' },
                  { name:'NODIS Library', url:'https://nodis3.gsfc.nasa.gov', desc:'All NPDs and NPRs (NASA Directives)' },
                  { name:'NASA Lessons Learned (LLIS)', url:'https://llis.nasa.gov', desc:'Vetted lessons learned from NASA programs and projects' },
                  { name:'NASA Software Engineering Handbook', url:'https://swehb.nasa.gov', desc:'Wiki-based NASA-HDBK-2203 implementation guidance' },
                  { name:'NASA Technical Reports Server (NTRS)', url:'https://ntrs.nasa.gov', desc:'Research papers, technical reports, and NASA publications' },
                  { name:'Space Industry Technical Standards', url:'https://space.commerce.gov/space-industry-technical-standards/', desc:'Office of Space Commerce Standards Compendium (2024)' },
                ].map(r => (
                  <a key={r.name} href={r.url} target="_blank" rel="noopener noreferrer"
                    className="flex items-start gap-3 p-4 bg-white border border-[#dadce0] rounded-xl hover:border-[#0b3d91] hover:shadow-sm transition-all group">
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
                  <thead><tr className="bg-[#f8f9fa] text-left">{['Center','Code','Location','Key Roles'].map(h => <th key={h} className="px-3 py-2.5 font-semibold text-[#202124] border-b border-[#dadce0]">{h}</th>)}</tr></thead>
                  <tbody>
                    {[
                      ['Johnson Space Center','JSC','Houston, TX','Human spaceflight, crew systems, mission control, human rating'],
                      ['Kennedy Space Center','KSC','Cape Canaveral, FL','Launch operations, Commercial Crew Program'],
                      ['Marshall Space Flight Center','MSFC','Huntsville, AL','SLS, propulsion, structural standards'],
                      ['Goddard Space Flight Center','GSFC','Greenbelt, MD','Earth science, space science, NODIS host'],
                      ['Jet Propulsion Laboratory','JPL','Pasadena, CA','Deep space missions, Mars rovers, FFRDC (Caltech)'],
                      ['Langley Research Center','LaRC','Hampton, VA','Aeronautics, atmospheric science, systems analysis'],
                      ['Glenn Research Center','GRC','Cleveland, OH','Propulsion, power systems, communications'],
                      ['Ames Research Center','ARC','Moffett Field, CA','Computing, thermal protection, astrobiology'],
                      ['Stennis Space Center','SSC','Bay St. Louis, MS','Rocket engine testing (RS-25, SLS)'],
                      ['Armstrong Flight Research','AFRC','Edwards AFB, CA','Flight research and testing'],
                    ].map((row, i) => (
                      <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-[#f8f9fa]'}>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-medium text-[#202124] text-xs">{row[0]}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] font-mono text-[#0b3d91] text-xs">{row[1]}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row[2]}</td>
                        <td className="px-3 py-2.5 border-b border-[#f1f3f4] text-xs text-[#5f6368]">{row[3]}</td>
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
