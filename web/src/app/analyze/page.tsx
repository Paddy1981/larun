'use client';

import { useState } from 'react';
import { Button, Card, Input } from '@/components/ui';
import { Header, Footer } from '@/components/layout';

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
    vetting: {
      disposition: string;
      odd_even: { flag: string; message: string };
      v_shape: { flag: string; message: string };
      secondary_eclipse: { flag: string; message: string };
    };
  };
}

export default function AnalyzePage() {
  const [ticId, setTicId] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!ticId.trim()) {
      setError('Please enter a TIC ID');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Submit analysis
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tic_id: ticId }),
      });

      if (!response.ok) {
        throw new Error('Failed to start analysis');
      }

      const { analysis_id } = await response.json();

      // Poll for results
      let attempts = 0;
      const maxAttempts = 60; // 5 minutes max

      while (attempts < maxAttempts) {
        const statusResponse = await fetch(`/api/v1/analyze/${analysis_id}`);
        const statusData = await statusResponse.json();

        if (statusData.status === 'completed' || statusData.status === 'failed') {
          setResult(statusData);
          break;
        }

        await new Promise(resolve => setTimeout(resolve, 5000));
        attempts++;
      }

      if (attempts >= maxAttempts) {
        setError('Analysis timed out. Please try again.');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-900">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-white mb-2">Analyze Target</h1>
          <p className="text-gray-400 mb-8">
            Enter a TESS Input Catalog (TIC) ID to search for exoplanet transit signals.
          </p>

          {/* Search Form */}
          <Card className="p-6 mb-8">
            <div className="flex gap-4">
              <Input
                type="text"
                placeholder="Enter TIC ID (e.g., 470710327)"
                value={ticId}
                onChange={(e) => setTicId(e.target.value)}
                className="flex-1"
                disabled={isLoading}
              />
              <Button
                onClick={handleAnalyze}
                disabled={isLoading || !ticId.trim()}
              >
                {isLoading ? 'Analyzing...' : 'Analyze'}
              </Button>
            </div>

            {error && (
              <p className="mt-4 text-red-400">{error}</p>
            )}
          </Card>

          {/* Loading State */}
          {isLoading && (
            <Card className="p-8 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
              <p className="text-white">Analyzing TIC {ticId}...</p>
              <p className="text-gray-400 text-sm mt-2">
                This may take up to 2 minutes. We're searching for transit signals,
                running vetting tests, and generating results.
              </p>
            </Card>
          )}

          {/* Results */}
          {result && result.status === 'completed' && result.result && (
            <div className="space-y-6">
              {/* Detection Result */}
              <Card className="p-6">
                <h2 className="text-xl font-semibold text-white mb-4">Detection Result</h2>

                <div className={`inline-block px-4 py-2 rounded-full text-lg font-bold mb-4 ${
                  result.result.detection
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-gray-500/20 text-gray-400'
                }`}>
                  {result.result.detection ? '🌍 Planet Candidate Detected!' : 'No Transit Detected'}
                </div>

                {result.result.detection && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    <div className="bg-gray-800 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Confidence</p>
                      <p className="text-2xl font-bold text-white">
                        {(result.result.confidence * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Period</p>
                      <p className="text-2xl font-bold text-white">
                        {result.result.period_days?.toFixed(4)} days
                      </p>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Depth</p>
                      <p className="text-2xl font-bold text-white">
                        {result.result.depth_ppm?.toFixed(0)} ppm
                      </p>
                    </div>
                    <div className="bg-gray-800 p-4 rounded-lg">
                      <p className="text-gray-400 text-sm">Duration</p>
                      <p className="text-2xl font-bold text-white">
                        {result.result.duration_hours?.toFixed(2)} hrs
                      </p>
                    </div>
                  </div>
                )}
              </Card>

              {/* Vetting Results */}
              {result.result.vetting && (
                <Card className="p-6">
                  <h2 className="text-xl font-semibold text-white mb-4">Vetting Results</h2>

                  <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium mb-4 ${
                    result.result.vetting.disposition === 'PLANET_CANDIDATE'
                      ? 'bg-green-500/20 text-green-400'
                      : result.result.vetting.disposition === 'LIKELY_FALSE_POSITIVE'
                      ? 'bg-red-500/20 text-red-400'
                      : 'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {result.result.vetting.disposition.replace(/_/g, ' ')}
                  </div>

                  <div className="space-y-3">
                    {/* Odd-Even Test */}
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <div>
                        <p className="text-white font-medium">Odd-Even Depth Test</p>
                        <p className="text-gray-400 text-sm">{result.result.vetting.odd_even.message}</p>
                      </div>
                      <span className={`px-2 py-1 rounded text-sm ${
                        result.result.vetting.odd_even.flag === 'PASS'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {result.result.vetting.odd_even.flag}
                      </span>
                    </div>

                    {/* V-Shape Test */}
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <div>
                        <p className="text-white font-medium">V-Shape Detection</p>
                        <p className="text-gray-400 text-sm">{result.result.vetting.v_shape.message}</p>
                      </div>
                      <span className={`px-2 py-1 rounded text-sm ${
                        result.result.vetting.v_shape.flag === 'PASS'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {result.result.vetting.v_shape.flag}
                      </span>
                    </div>

                    {/* Secondary Eclipse */}
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <div>
                        <p className="text-white font-medium">Secondary Eclipse</p>
                        <p className="text-gray-400 text-sm">{result.result.vetting.secondary_eclipse.message}</p>
                      </div>
                      <span className={`px-2 py-1 rounded text-sm ${
                        result.result.vetting.secondary_eclipse.flag === 'PASS'
                          ? 'bg-green-500/20 text-green-400'
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        {result.result.vetting.secondary_eclipse.flag}
                      </span>
                    </div>
                  </div>
                </Card>
              )}

              {/* Actions */}
              <div className="flex gap-4">
                <Button variant="outline" onClick={() => setResult(null)}>
                  Analyze Another Target
                </Button>
                <Button>
                  Save to Dashboard
                </Button>
              </div>
            </div>
          )}

          {/* Error State */}
          {result && result.status === 'failed' && (
            <Card className="p-6 border-red-500/50">
              <h2 className="text-xl font-semibold text-red-400 mb-2">Analysis Failed</h2>
              <p className="text-gray-400">
                Unable to analyze TIC {ticId}. This could be due to insufficient data
                or the target not being observed by TESS.
              </p>
              <Button className="mt-4" onClick={() => setResult(null)}>
                Try Again
              </Button>
            </Card>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
