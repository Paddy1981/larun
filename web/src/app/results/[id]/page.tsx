'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { Button, Card } from '@/components/ui';
import { Header, Footer } from '@/components/layout';

interface AnalysisResult {
  id: string;
  tic_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  result?: {
    detection: boolean;
    confidence: number;
    period_days: number | null;
    depth_ppm: number | null;
    duration_hours: number | null;
    epoch_btjd: number | null;
    snr: number | null;
    vetting: {
      disposition: string;
      odd_even_passed: boolean;
      odd_even_sigma: number;
      secondary_passed: boolean;
      secondary_depth_ratio: number;
      v_shape_passed: boolean;
      v_shape_metric: number;
    };
  };
  error_message?: string;
}

export default function ResultsPage() {
  const params = useParams();
  const analysisId = params.id as string;

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      try {
        const res = await fetch(`/api/v1/analyses/${analysisId}`);
        if (!res.ok) {
          throw new Error('Analysis not found');
        }
        const data = await res.json();
        setAnalysis(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load analysis');
      } finally {
        setIsLoading(false);
      }
    };

    if (analysisId) {
      fetchAnalysis();
    }
  }, [analysisId]);

  const getDispositionColor = (disposition: string) => {
    if (disposition === 'PLANET_CANDIDATE') return 'text-green-400';
    if (disposition === 'LIKELY_FALSE_POSITIVE') return 'text-red-400';
    return 'text-yellow-400';
  };

  const getTestBadge = (passed: boolean) => {
    return passed
      ? 'bg-green-500/20 text-green-400'
      : 'bg-red-500/20 text-red-400';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex flex-col bg-gray-900">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
            <p className="text-gray-400 mt-4">Loading analysis results...</p>
          </div>
        </main>
        <Footer />
      </div>
    );
  }

  if (error || !analysis) {
    return (
      <div className="min-h-screen flex flex-col bg-gray-900">
        <Header />
        <main className="flex-1 flex items-center justify-center">
          <Card className="p-8 text-center max-w-md">
            <div className="text-6xl mb-4">❌</div>
            <h2 className="text-xl font-semibold text-white mb-2">Analysis Not Found</h2>
            <p className="text-gray-400 mb-6">{error || 'The requested analysis could not be found.'}</p>
            <Link href="/dashboard">
              <Button>Back to Dashboard</Button>
            </Link>
          </Card>
        </main>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-900">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="flex justify-between items-start mb-8">
            <div>
              <Link href="/dashboard" className="text-blue-400 hover:underline text-sm mb-2 inline-block">
                ← Back to Dashboard
              </Link>
              <h1 className="text-3xl font-bold text-white">TIC {analysis.tic_id}</h1>
              <p className="text-gray-400">
                Analysis submitted {new Date(analysis.created_at).toLocaleDateString()}
                {analysis.completed_at && ` • Completed ${new Date(analysis.completed_at).toLocaleDateString()}`}
              </p>
            </div>
            <span className={`px-3 py-1 rounded text-sm ${
              analysis.status === 'completed' ? 'bg-green-500/20 text-green-400' :
              analysis.status === 'processing' ? 'bg-blue-500/20 text-blue-400' :
              analysis.status === 'failed' ? 'bg-red-500/20 text-red-400' :
              'bg-yellow-500/20 text-yellow-400'
            }`}>
              {analysis.status.toUpperCase()}
            </span>
          </div>

          {analysis.status === 'processing' && (
            <Card className="p-8 text-center mb-6">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
              <p className="text-gray-400 mt-4">Analysis in progress...</p>
              <p className="text-gray-500 text-sm mt-2">This typically takes 30-60 seconds</p>
            </Card>
          )}

          {analysis.status === 'failed' && (
            <Card className="p-6 mb-6 bg-red-900/20 border-red-500/30">
              <h3 className="text-lg font-semibold text-red-400 mb-2">Analysis Failed</h3>
              <p className="text-gray-300">{analysis.error_message || 'An unknown error occurred during analysis.'}</p>
            </Card>
          )}

          {analysis.status === 'completed' && analysis.result && (
            <>
              {/* Detection Summary */}
              <Card className={`p-6 mb-6 ${
                analysis.result.vetting.disposition === 'PLANET_CANDIDATE'
                  ? 'bg-green-900/20 border-green-500/30'
                  : analysis.result.vetting.disposition === 'LIKELY_FALSE_POSITIVE'
                  ? 'bg-red-900/20 border-red-500/30'
                  : 'bg-yellow-900/20 border-yellow-500/30'
              }`}>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-gray-400 text-sm mb-1">Disposition</p>
                    <p className={`text-2xl font-bold ${getDispositionColor(analysis.result.vetting.disposition)}`}>
                      {analysis.result.vetting.disposition.replace(/_/g, ' ')}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-gray-400 text-sm mb-1">Confidence</p>
                    <p className="text-2xl font-bold text-white">
                      {(analysis.result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </Card>

              {/* Transit Parameters */}
              {analysis.result.detection && (
                <Card className="p-6 mb-6">
                  <h2 className="text-xl font-semibold text-white mb-4">Transit Parameters</h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-gray-400 text-sm">Period</p>
                      <p className="text-xl font-mono text-white">
                        {analysis.result.period_days?.toFixed(6) || '-'} d
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Depth</p>
                      <p className="text-xl font-mono text-white">
                        {analysis.result.depth_ppm?.toFixed(0) || '-'} ppm
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Duration</p>
                      <p className="text-xl font-mono text-white">
                        {analysis.result.duration_hours?.toFixed(2) || '-'} h
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">SNR</p>
                      <p className="text-xl font-mono text-white">
                        {analysis.result.snr?.toFixed(1) || '-'}
                      </p>
                    </div>
                    <div>
                      <p className="text-gray-400 text-sm">Epoch (BTJD)</p>
                      <p className="text-xl font-mono text-white">
                        {analysis.result.epoch_btjd?.toFixed(4) || '-'}
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              {/* Vetting Results */}
              <Card className="p-6 mb-6">
                <h2 className="text-xl font-semibold text-white mb-4">Vetting Results</h2>
                <div className="space-y-4">
                  {/* Odd-Even Test */}
                  <div className="flex items-center justify-between p-4 bg-gray-800 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">Odd-Even Depth Test</h3>
                      <p className="text-gray-400 text-sm">
                        Compares transit depths of odd and even transits
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`px-3 py-1 rounded text-sm ${getTestBadge(analysis.result.vetting.odd_even_passed)}`}>
                        {analysis.result.vetting.odd_even_passed ? 'PASSED' : 'FAILED'}
                      </span>
                      <p className="text-gray-400 text-sm mt-1">
                        σ = {analysis.result.vetting.odd_even_sigma.toFixed(2)}
                      </p>
                    </div>
                  </div>

                  {/* Secondary Eclipse Test */}
                  <div className="flex items-center justify-between p-4 bg-gray-800 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">Secondary Eclipse Test</h3>
                      <p className="text-gray-400 text-sm">
                        Checks for eclipsing binary signature at phase 0.5
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`px-3 py-1 rounded text-sm ${getTestBadge(analysis.result.vetting.secondary_passed)}`}>
                        {analysis.result.vetting.secondary_passed ? 'PASSED' : 'FAILED'}
                      </span>
                      <p className="text-gray-400 text-sm mt-1">
                        Depth ratio: {analysis.result.vetting.secondary_depth_ratio.toFixed(2)}
                      </p>
                    </div>
                  </div>

                  {/* V-Shape Test */}
                  <div className="flex items-center justify-between p-4 bg-gray-800 rounded-lg">
                    <div>
                      <h3 className="text-white font-medium">V-Shape Test</h3>
                      <p className="text-gray-400 text-sm">
                        Detects grazing eclipsing binaries from transit shape
                      </p>
                    </div>
                    <div className="text-right">
                      <span className={`px-3 py-1 rounded text-sm ${getTestBadge(analysis.result.vetting.v_shape_passed)}`}>
                        {analysis.result.vetting.v_shape_passed ? 'PASSED' : 'FAILED'}
                      </span>
                      <p className="text-gray-400 text-sm mt-1">
                        Metric: {analysis.result.vetting.v_shape_metric.toFixed(2)}
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Actions */}
              <Card className="p-6">
                <h2 className="text-xl font-semibold text-white mb-4">Actions</h2>
                <div className="flex flex-wrap gap-4">
                  <Button variant="secondary">
                    Download Report (PDF)
                  </Button>
                  <Button variant="secondary">
                    Export Data (CSV)
                  </Button>
                  <Button variant="secondary">
                    View on ExoFOP
                  </Button>
                </div>
              </Card>
            </>
          )}

          {!analysis.result?.detection && analysis.status === 'completed' && (
            <Card className="p-8 text-center">
              <div className="text-6xl mb-4">🔍</div>
              <h2 className="text-xl font-semibold text-white mb-2">No Transit Detected</h2>
              <p className="text-gray-400 mb-6">
                The analysis did not detect a statistically significant transit signal in this light curve.
              </p>
              <Link href="/analyze">
                <Button>Try Another Target</Button>
              </Link>
            </Card>
          )}
        </div>
      </main>

      <Footer />
    </div>
  );
}
