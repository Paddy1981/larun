'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

interface Analysis {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    vetting: {
      disposition: string;
    };
  };
}

interface StatsData {
  objectsProcessed: number;
  detections: number;
  modelAccuracy: number;
  vettedCandidates: number;
  sessionDuration: string;
}

// Sample TESS Objects of Interest
const tessTargets = [
  { id: 'TIC 307210830', toi: 'TOI-700', period: '37.42', depth: '0.82%', mag: '13.1', priority: 95 },
  { id: 'TIC 470710327', toi: 'TOI-1338', period: '14.61', depth: '0.24%', mag: '11.7', priority: 88 },
  { id: 'TIC 441462736', toi: 'TOI-849', period: '18.36', depth: '0.31%', mag: '12.8', priority: 82 },
  { id: 'TIC 141527579', toi: 'TOI-561', period: '10.78', depth: '0.19%', mag: '10.3', priority: 79 },
  { id: 'TIC 231702397', toi: 'TOI-1231', period: '24.25', depth: '0.45%', mag: '12.3', priority: 75 },
];

export default function DashboardPage() {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [stats, setStats] = useState<StatsData>({
    objectsProcessed: 0,
    detections: 0,
    modelAccuracy: 81.8,
    vettedCandidates: 0,
    sessionDuration: '0h 0m',
  });
  const [isLoading, setIsLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const analysesRes = await fetch('/api/v1/analyses');
        if (analysesRes.ok) {
          const data = await analysesRes.json();
          setAnalyses(data.analyses || []);

          // Calculate stats from analyses
          const completed = data.analyses?.filter((a: Analysis) => a.status === 'completed') || [];
          const detections = completed.filter((a: Analysis) => a.result?.detection);
          const candidates = completed.filter((a: Analysis) => a.result?.vetting?.disposition === 'PLANET_CANDIDATE');

          setStats({
            objectsProcessed: completed.length,
            detections: detections.length,
            modelAccuracy: 81.8,
            vettedCandidates: candidates.length,
            sessionDuration: '1h 23m',
          });
        }
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const recentActivity = [
    { type: 'detection', message: 'Transit signal detected in TIC 307210830', time: '2 min ago' },
    { type: 'vetting', message: 'Vetting completed for TOI-700 b', time: '15 min ago' },
    { type: 'calibration', message: 'Light curve calibrated for TIC 470710327', time: '1 hour ago' },
    { type: 'detection', message: 'New analysis started for TIC 141527579', time: '2 hours ago' },
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0a]">
      {/* Top Navigation */}
      <header className="fixed top-0 left-0 right-0 h-16 bg-[#0a0a0a]/95 backdrop-blur-md border-b border-gray-800 z-50">
        <div className="flex items-center justify-between h-full px-4">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-gray-800 rounded-lg lg:hidden">
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            <Link href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 bg-gradient-to-br from-[#6366f1] to-[#3b82f6] rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="3" />
                </svg>
              </div>
              <span className="text-lg font-semibold text-white">LARUN</span>
            </Link>
          </div>

          {/* Center Nav */}
          <nav className="hidden md:flex items-center gap-6">
            <Link href="/" className="text-gray-400 hover:text-white text-sm transition-colors">Home</Link>
            <Link href="/dashboard" className="text-white text-sm font-medium">Dashboard</Link>
            <Link href="/#pricing" className="text-gray-400 hover:text-white text-sm transition-colors">Pricing</Link>
            <Link href="/#features" className="text-gray-400 hover:text-white text-sm transition-colors">Docs</Link>
          </nav>

          {/* Right Icons */}
          <div className="flex items-center gap-2">
            <button className="p-2 hover:bg-gray-800 rounded-lg">
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
            </button>
            <button className="p-2 hover:bg-gray-800 rounded-lg">
              <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
            <div className="w-8 h-8 bg-[#6366f1] rounded-full flex items-center justify-center text-white text-sm font-medium ml-2">
              U
            </div>
          </div>
        </div>
      </header>

      {/* Sidebar */}
      <aside className={`fixed top-16 left-0 bottom-0 w-64 bg-[#121212] border-r border-gray-800 overflow-y-auto transition-transform z-40 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0`}>
        <div className="p-4">
          {/* Products Section */}
          <div className="mb-6">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">Products</p>
            <nav className="space-y-1">
              <Link href="/analyze" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#6366f1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Pipeline
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#3b82f6]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
                Calibration
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#10b981]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
                Detector
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#f59e0b]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Vetting
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#ef4444]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Reports
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-[#8b5cf6]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                Model
              </Link>
            </nav>
          </div>

          {/* Interactive Section */}
          <div className="mb-6">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">Interactive</p>
            <nav className="space-y-1">
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                Web Terminal
              </Link>
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                AI Chat
              </Link>
            </nav>
          </div>

          {/* Account Section */}
          <div>
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-3">Account</p>
            <nav className="space-y-1">
              <Link href="#" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Profile
              </Link>
              <Link href="/settings/subscription" className="flex items-center gap-3 px-3 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Usage & Billing
              </Link>
            </nav>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="lg:ml-64 pt-16 min-h-screen">
        <div className="p-6">
          {/* Hero Card */}
          <div className="bg-gradient-to-r from-[#6366f1]/20 to-[#3b82f6]/20 border border-[#6366f1]/30 rounded-xl p-6 mb-6">
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
              <div>
                <h1 className="text-2xl font-bold text-white mb-2">Welcome to LARUN Dashboard</h1>
                <p className="text-gray-400">TinyML-powered spectral data analysis for exoplanet detection. Analyze NASA TESS and Kepler light curves with 81.8% accuracy.</p>
              </div>
              <div className="flex gap-3">
                <Link href="/analyze" className="bg-[#6366f1] hover:bg-[#5558e3] text-white font-medium px-4 py-2 rounded-lg transition-all hover:shadow-lg hover:shadow-[#6366f1]/25">
                  Run Analysis
                </Link>
                <Link href="/#features" className="bg-gray-800 hover:bg-gray-700 text-white font-medium px-4 py-2 rounded-lg border border-gray-700 transition-colors">
                  View Docs
                </Link>
              </div>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-[#121212] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">Objects Processed</span>
                <svg className="w-5 h-5 text-[#6366f1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{stats.objectsProcessed}</p>
              <p className="text-gray-500 text-xs mt-1">Session: {stats.sessionDuration}</p>
            </div>

            <div className="bg-[#121212] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">Detections</span>
                <svg className="w-5 h-5 text-[#10b981]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{stats.detections}</p>
              <p className="text-gray-500 text-xs mt-1">Transit signals found</p>
            </div>

            <div className="bg-[#121212] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">Model Accuracy</span>
                <svg className="w-5 h-5 text-[#fbbf24]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-[#fbbf24]">{stats.modelAccuracy}%</p>
              <p className="text-gray-500 text-xs mt-1">TinyML precision</p>
            </div>

            <div className="bg-[#121212] border border-gray-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-gray-400 text-sm">Vetted Candidates</span>
                <svg className="w-5 h-5 text-[#8b5cf6]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                </svg>
              </div>
              <p className="text-3xl font-bold text-white">{stats.vettedCandidates}</p>
              <p className="text-gray-500 text-xs mt-1">Planet candidates</p>
            </div>
          </div>

          {/* Target Discovery Table */}
          <div className="bg-[#121212] border border-gray-800 rounded-xl p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-white">TESS Objects of Interest</h2>
              <button className="text-gray-400 hover:text-white p-2 hover:bg-gray-800 rounded-lg transition-colors">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-gray-400 border-b border-gray-800">
                    <th className="pb-3 font-medium text-sm">Target ID</th>
                    <th className="pb-3 font-medium text-sm">TOI</th>
                    <th className="pb-3 font-medium text-sm">Period (days)</th>
                    <th className="pb-3 font-medium text-sm">Depth</th>
                    <th className="pb-3 font-medium text-sm">Mag</th>
                    <th className="pb-3 font-medium text-sm">Priority</th>
                    <th className="pb-3 font-medium text-sm">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {tessTargets.map((target, index) => (
                    <tr key={index} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                      <td className="py-3 text-white font-mono text-sm">{target.id}</td>
                      <td className="py-3 text-[#6366f1] text-sm">{target.toi}</td>
                      <td className="py-3 text-gray-300 text-sm">{target.period}</td>
                      <td className="py-3 text-gray-300 text-sm">{target.depth}</td>
                      <td className="py-3 text-gray-300 text-sm">{target.mag}</td>
                      <td className="py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 bg-gray-700 rounded-full h-1.5">
                            <div className="bg-[#10b981] h-1.5 rounded-full" style={{ width: `${target.priority}%` }}></div>
                          </div>
                          <span className="text-gray-400 text-xs">{target.priority}%</span>
                        </div>
                      </td>
                      <td className="py-3">
                        <Link href={`/analyze?tic=${target.id.replace('TIC ', '')}`} className="text-[#6366f1] hover:text-[#818cf8] text-sm font-medium">
                          Analyze
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Two Column Layout */}
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Recent Analyses */}
            <div className="bg-[#121212] border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Recent Analyses</h2>

              {isLoading ? (
                <div className="text-center py-8">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#6366f1] mx-auto"></div>
                  <p className="text-gray-400 mt-2 text-sm">Loading analyses...</p>
                </div>
              ) : analyses.length === 0 ? (
                <div className="text-center py-8">
                  <div className="w-12 h-12 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-3">
                    <svg className="w-6 h-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <p className="text-gray-400 mb-3 text-sm">No analyses yet</p>
                  <Link href="/analyze" className="text-[#6366f1] hover:text-[#818cf8] text-sm font-medium">
                    Start your first analysis →
                  </Link>
                </div>
              ) : (
                <div className="space-y-3">
                  {analyses.slice(0, 5).map((analysis) => (
                    <Link key={analysis.id} href={`/results/${analysis.id}`} className="block p-3 bg-gray-800/30 hover:bg-gray-800/50 rounded-lg transition-colors">
                      <div className="flex items-center justify-between">
                        <span className="text-white font-mono text-sm">TIC {analysis.tic_id}</span>
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          analysis.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          analysis.status === 'processing' ? 'bg-blue-500/20 text-blue-400' :
                          analysis.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                          'bg-yellow-500/20 text-yellow-400'
                        }`}>
                          {analysis.status}
                        </span>
                      </div>
                      <p className="text-gray-500 text-xs mt-1">
                        {new Date(analysis.created_at).toLocaleDateString()}
                      </p>
                    </Link>
                  ))}
                </div>
              )}
            </div>

            {/* Activity Feed */}
            <div className="bg-[#121212] border border-gray-800 rounded-xl p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-white">Recent Activity</h2>
                <button className="text-gray-400 hover:text-white p-2 hover:bg-gray-800 rounded-lg transition-colors">
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                {recentActivity.map((activity, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                      activity.type === 'detection' ? 'bg-[#10b981]/20' :
                      activity.type === 'vetting' ? 'bg-[#f59e0b]/20' :
                      'bg-[#6366f1]/20'
                    }`}>
                      <svg className={`w-4 h-4 ${
                        activity.type === 'detection' ? 'text-[#10b981]' :
                        activity.type === 'vetting' ? 'text-[#f59e0b]' :
                        'text-[#6366f1]'
                      }`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        {activity.type === 'detection' ? (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                        ) : activity.type === 'vetting' ? (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
                        ) : (
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                        )}
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-300 text-sm">{activity.message}</p>
                      <p className="text-gray-500 text-xs mt-1">{activity.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
