'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useSession } from 'next-auth/react';
import Header from '@/components/Header';

interface AnalysisResult {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    depth_ppm: number | null;
    duration_hours: number | null;
    vetting?: {
      disposition: string;
      odd_even: { flag: string; message: string };
      v_shape: { flag: string; message: string };
      secondary_eclipse: { flag: string; message: string };
    };
  };
  error?: string;
}

const popularTargets = [
  { id: '470710327', name: 'TOI-1338 b', description: 'Circumbinary planet' },
  { id: '307210830', name: 'TOI-700 d', description: 'Earth-sized in HZ' },
  { id: '261136679', name: 'TOI-175 b', description: 'Super-Earth' },
];

export default function AnalyzePage() {
  const { data: session, status } = useSession();
  const [ticId, setTicId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (status === 'unauthenticated') {
      window.location.href = '/cloud/auth/login?redirect=/analyze';
    }
  }, [status]);

  const handleAnalyze = async (targetId?: string) => {
    const idToAnalyze = targetId || ticId.trim();
    if (!idToAnalyze) {
      setError('Please enter a TIC ID');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setProgress(0);

    try {
      // Simulate progress while waiting for synchronous result
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(85, prev + 5));
      }, 1500);

      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tic_id: idToAnalyze }),
      });

      clearInterval(progressInterval);

      // Safely parse — Vercel can return plain-text on timeout/crash
      const text = await response.text();
      let data: Record<string, unknown>;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(
          response.status === 504
            ? 'Analysis timed out fetching TESS data. Try again or pick a different target.'
            : `Server error (${response.status}). Please try again.`
        );
      }

      if (!response.ok) {
        throw new Error((data.error as { message?: string })?.message || 'Failed to start analysis');
      }

      setProgress(100);
      setResult({
        id: data.analysis_id as string,
        tic_id: data.tic_id as string,
        status: data.status as AnalysisResult['status'],
        result: data.result as AnalysisResult['result'],
        error: data.error as string | undefined,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isLoading && ticId.trim()) {
      handleAnalyze();
    }
  };

  // Show loading while checking authentication
  if (status === 'loading') {
    return (
      <div className="min-h-screen flex flex-col bg-white">
        <Header />
        <main className="flex-1 pt-24 pb-16 flex items-center justify-center">
          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-4 relative">
              <div className="absolute inset-0 border-4 border-[#f1f3f4] rounded-full"></div>
              <div className="absolute inset-0 border-4 border-[#1a73e8] rounded-full border-t-transparent animate-spin"></div>
            </div>
            <p className="text-[#5f6368]">Loading...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <Header />

      <main className="flex-1 pt-24 pb-16">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Title */}
          <div className="text-center mb-10">
            <h1 className="text-3xl font-bold text-[#202124] mb-3">Analyze Target</h1>
            <p className="text-[#5f6368] max-w-2xl mx-auto">
              Enter a TESS Input Catalog (TIC) ID to search for exoplanet transit signals using our TinyML detection models.
            </p>
          </div>

          {/* Search Form */}
          <div className="bg-white border border-[#dadce0] rounded-xl p-6 mb-6 shadow-sm">
            <div className="flex gap-3">
              <input
                type="text"
                placeholder="Enter TIC ID (e.g., 470710327)"
                value={ticId}
                onChange={(e) => setTicId(e.target.value)}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="flex-1 px-4 py-3 border border-[#dadce0] rounded-lg text-[#202124] placeholder-[#5f6368] focus:outline-none focus:ring-2 focus:ring-[#1a73e8] focus:border-transparent disabled:bg-[#f1f3f4] disabled:cursor-not-allowed"
              />
              <button
                onClick={() => handleAnalyze()}
                disabled={isLoading || !ticId.trim()}
                className="px-6 py-3 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors disabled:bg-[#dadce0] disabled:cursor-not-allowed"
              >
                {isLoading ? 'Analyzing...' : 'Analyze'}
              </button>
            </div>

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                {error}
              </div>
            )}
          </div>

          {/* Popular Targets */}
          {!isLoading && !result && (
            <div className="mb-8">
              <p className="text-sm text-[#5f6368] mb-3">Try a known target:</p>
              <div className="flex flex-wrap gap-2">
                {popularTargets.map((target) => (
                  <button
                    key={target.id}
                    onClick={() => {
                      setTicId(target.id);
                      handleAnalyze(target.id);
                    }}
                    className="px-4 py-2 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] text-sm rounded-full transition-colors"
                  >
                    TIC {target.id} ({target.name})
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Loading State */}
          {isLoading && (
            <div className="bg-white border border-[#dadce0] rounded-xl p-8 text-center shadow-sm">
              <div className="w-16 h-16 mx-auto mb-4 relative">
                <div className="absolute inset-0 border-4 border-[#f1f3f4] rounded-full"></div>
                <div
                  className="absolute inset-0 border-4 border-[#1a73e8] rounded-full border-t-transparent animate-spin"
                ></div>
              </div>
              <h3 className="text-lg font-medium text-[#202124] mb-2">Analyzing TIC {ticId || 'target'}...</h3>
              <p className="text-[#5f6368] text-sm mb-4">
                Running transit detection and vetting tests
              </p>
              <div className="w-full bg-[#f1f3f4] rounded-full h-2 mb-2">
                <div
                  className="bg-[#1a73e8] h-2 rounded-full transition-all duration-500"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-xs text-[#5f6368]">{progress}% complete</p>
            </div>
          )}

          {/* Results - Detection Found */}
          {result && result.status === 'completed' && result.result && (
            <div className="space-y-6">
              {/* Detection Summary */}
              <div className={`rounded-xl p-6 ${
                result.result.detection
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-[#f1f3f4] border border-[#dadce0]'
              }`}>
                <div className="flex items-center gap-4 mb-4">
                  <div className={`w-14 h-14 rounded-full flex items-center justify-center ${
                    result.result.detection ? 'bg-green-100' : 'bg-[#dadce0]'
                  }`}>
                    {result.result.detection ? (
                      <svg className="w-8 h-8 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    ) : (
                      <svg className="w-8 h-8 text-[#5f6368]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    )}
                  </div>
                  <div>
                    <h2 className={`text-xl font-bold ${result.result.detection ? 'text-green-700' : 'text-[#202124]'}`}>
                      {result.result.detection ? 'Planet Candidate Detected!' : 'No Transit Signal Detected'}
                    </h2>
                    <p className="text-[#5f6368]">TIC {result.tic_id}</p>
                  </div>
                </div>

                {result.result.detection && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <p className="text-xs text-[#5f6368] uppercase tracking-wider mb-1">Confidence</p>
                      <p className="text-2xl font-bold text-[#202124]">
                        {(result.result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <p className="text-xs text-[#5f6368] uppercase tracking-wider mb-1">Period</p>
                      <p className="text-2xl font-bold text-[#202124]">
                        {result.result.period_days?.toFixed(2)} <span className="text-sm font-normal">days</span>
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <p className="text-xs text-[#5f6368] uppercase tracking-wider mb-1">Depth</p>
                      <p className="text-2xl font-bold text-[#202124]">
                        {result.result.depth_ppm?.toFixed(0)} <span className="text-sm font-normal">ppm</span>
                      </p>
                    </div>
                    <div className="bg-white rounded-lg p-4 border border-green-200">
                      <p className="text-xs text-[#5f6368] uppercase tracking-wider mb-1">Duration</p>
                      <p className="text-2xl font-bold text-[#202124]">
                        {result.result.duration_hours?.toFixed(1)} <span className="text-sm font-normal">hrs</span>
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Vetting Results */}
              {result.result.vetting && (
                <div className="bg-white border border-[#dadce0] rounded-xl p-6 shadow-sm">
                  <h3 className="text-lg font-semibold text-[#202124] mb-4">Vetting Results</h3>

                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-4 ${
                    result.result.vetting.disposition === 'PLANET_CANDIDATE'
                      ? 'bg-green-100 text-green-700'
                      : result.result.vetting.disposition === 'LIKELY_FALSE_POSITIVE'
                      ? 'bg-red-100 text-red-700'
                      : 'bg-yellow-100 text-yellow-700'
                  }`}>
                    {result.result.vetting.disposition.replace(/_/g, ' ')}
                  </div>

                  <div className="space-y-3">
                    {/* Odd-Even Test */}
                    <div className="flex items-center justify-between p-4 bg-[#f8f9fa] rounded-lg">
                      <div>
                        <p className="font-medium text-[#202124]">Odd-Even Depth Test</p>
                        <p className="text-sm text-[#5f6368]">{result.result.vetting.odd_even.message}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        result.result.vetting.odd_even.flag === 'PASS'
                          ? 'bg-green-100 text-green-700'
                          : result.result.vetting.odd_even.flag === 'WARNING'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {result.result.vetting.odd_even.flag}
                      </span>
                    </div>

                    {/* V-Shape Test */}
                    <div className="flex items-center justify-between p-4 bg-[#f8f9fa] rounded-lg">
                      <div>
                        <p className="font-medium text-[#202124]">V-Shape Analysis</p>
                        <p className="text-sm text-[#5f6368]">{result.result.vetting.v_shape.message}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        result.result.vetting.v_shape.flag === 'PASS'
                          ? 'bg-green-100 text-green-700'
                          : result.result.vetting.v_shape.flag === 'WARNING'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {result.result.vetting.v_shape.flag}
                      </span>
                    </div>

                    {/* Secondary Eclipse */}
                    <div className="flex items-center justify-between p-4 bg-[#f8f9fa] rounded-lg">
                      <div>
                        <p className="font-medium text-[#202124]">Secondary Eclipse Check</p>
                        <p className="text-sm text-[#5f6368]">{result.result.vetting.secondary_eclipse.message}</p>
                      </div>
                      <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                        result.result.vetting.secondary_eclipse.flag === 'PASS'
                          ? 'bg-green-100 text-green-700'
                          : result.result.vetting.secondary_eclipse.flag === 'WARNING'
                          ? 'bg-yellow-100 text-yellow-700'
                          : 'bg-red-100 text-red-700'
                      }`}>
                        {result.result.vetting.secondary_eclipse.flag}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-4">
                <button
                  onClick={() => {
                    setResult(null);
                    setTicId('');
                  }}
                  className="px-6 py-3 bg-[#f1f3f4] hover:bg-[#e8eaed] text-[#202124] font-medium rounded-lg transition-colors"
                >
                  Analyze Another
                </button>
                <Link
                  href="/dashboard"
                  className="px-6 py-3 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors"
                >
                  View Dashboard
                </Link>
              </div>
            </div>
          )}

          {/* Error State */}
          {result && result.status === 'failed' && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-red-700 mb-2">Analysis Failed</h3>
              <p className="text-[#5f6368] mb-4">
                {result.error || `Unable to analyze TIC ${result.tic_id}. This could be due to insufficient data or the target not being observed by TESS.`}
              </p>
              <button
                onClick={() => {
                  setResult(null);
                  setTicId('');
                }}
                className="px-6 py-3 bg-[#1a73e8] hover:bg-[#1557b0] text-white font-medium rounded-lg transition-colors"
              >
                Try Again
              </button>
            </div>
          )}

          {/* How It Works */}
          {!isLoading && !result && (
            <div className="mt-12 bg-[#f8f9fa] rounded-xl p-6">
              <h3 className="text-lg font-semibold text-[#202124] mb-4">How It Works</h3>
              <div className="grid md:grid-cols-3 gap-6">
                <div className="flex gap-3">
                  <div className="w-8 h-8 bg-[#1a73e8] text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">1</div>
                  <div>
                    <p className="font-medium text-[#202124]">Data Retrieval</p>
                    <p className="text-sm text-[#5f6368]">We fetch light curve data from NASA TESS archives</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="w-8 h-8 bg-[#1a73e8] text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">2</div>
                  <div>
                    <p className="font-medium text-[#202124]">Transit Detection</p>
                    <p className="text-sm text-[#5f6368]">TinyML models search for periodic dimming signals</p>
                  </div>
                </div>
                <div className="flex gap-3">
                  <div className="w-8 h-8 bg-[#1a73e8] text-white rounded-full flex items-center justify-center font-bold text-sm flex-shrink-0">3</div>
                  <div>
                    <p className="font-medium text-[#202124]">Vetting Tests</p>
                    <p className="text-sm text-[#5f6368]">Multiple tests rule out false positives</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <p className="text-sm text-[#5f6368]">
            &copy; {new Date().getFullYear()} Larun Engineering. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
