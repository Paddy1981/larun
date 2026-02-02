'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Button, Card } from '@/components/ui';
import { Header, Footer } from '@/components/layout';

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

interface UsageData {
  analyses_this_month: number;
  analyses_limit: number;
  period_start: string;
  period_end: string;
}

export default function DashboardPage() {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [usage, setUsage] = useState<UsageData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch user's analyses
    const fetchData = async () => {
      try {
        const [analysesRes, usageRes] = await Promise.all([
          fetch('/api/v1/analyses'),
          fetch('/api/v1/user/usage'),
        ]);

        if (analysesRes.ok) {
          const data = await analysesRes.json();
          setAnalyses(data.analyses || []);
        }

        if (usageRes.ok) {
          const data = await usageRes.json();
          setUsage(data);
        }
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, []);

  const getStatusBadge = (status: string) => {
    const styles: Record<string, string> = {
      completed: 'bg-green-500/20 text-green-400',
      processing: 'bg-blue-500/20 text-blue-400',
      pending: 'bg-yellow-500/20 text-yellow-400',
      failed: 'bg-red-500/20 text-red-400',
    };
    return styles[status] || 'bg-gray-500/20 text-gray-400';
  };

  const getDispositionBadge = (disposition: string) => {
    if (disposition === 'PLANET_CANDIDATE') {
      return 'bg-green-500/20 text-green-400';
    } else if (disposition === 'LIKELY_FALSE_POSITIVE') {
      return 'bg-red-500/20 text-red-400';
    }
    return 'bg-yellow-500/20 text-yellow-400';
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-900">
      <Header />

      <main className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <div>
              <h1 className="text-3xl font-bold text-white">Dashboard</h1>
              <p className="text-gray-400">Manage your exoplanet analyses</p>
            </div>
            <Link href="/analyze">
              <Button>New Analysis</Button>
            </Link>
          </div>

          {/* Usage Stats */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <Card className="p-6">
              <p className="text-gray-400 text-sm mb-1">Analyses This Month</p>
              <p className="text-3xl font-bold text-white">
                {usage?.analyses_this_month || 0}
                <span className="text-lg text-gray-500">
                  {' '}/ {usage?.analyses_limit || 25}
                </span>
              </p>
              <div className="mt-2 bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 rounded-full h-2"
                  style={{
                    width: `${((usage?.analyses_this_month || 0) / (usage?.analyses_limit || 25)) * 100}%`
                  }}
                />
              </div>
            </Card>

            <Card className="p-6">
              <p className="text-gray-400 text-sm mb-1">Planet Candidates</p>
              <p className="text-3xl font-bold text-green-400">
                {analyses.filter(a =>
                  a.result?.vetting?.disposition === 'PLANET_CANDIDATE'
                ).length}
              </p>
              <p className="text-gray-500 text-sm mt-1">
                From {analyses.filter(a => a.status === 'completed').length} completed analyses
              </p>
            </Card>

            <Card className="p-6">
              <p className="text-gray-400 text-sm mb-1">Subscription</p>
              <p className="text-xl font-bold text-white">Hobbyist</p>
              <p className="text-gray-500 text-sm mt-1">
                $9/month • 25 analyses
              </p>
              <Link href="/settings/subscription" className="text-blue-400 text-sm hover:underline">
                Manage subscription →
              </Link>
            </Card>
          </div>

          {/* Recent Analyses */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold text-white mb-4">Recent Analyses</h2>

            {isLoading ? (
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
                <p className="text-gray-400 mt-2">Loading analyses...</p>
              </div>
            ) : analyses.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">🔭</div>
                <p className="text-gray-400 mb-4">No analyses yet</p>
                <Link href="/analyze">
                  <Button>Start Your First Analysis</Button>
                </Link>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="text-left text-gray-400 border-b border-gray-700">
                      <th className="pb-3 font-medium">TIC ID</th>
                      <th className="pb-3 font-medium">Date</th>
                      <th className="pb-3 font-medium">Status</th>
                      <th className="pb-3 font-medium">Result</th>
                      <th className="pb-3 font-medium">Confidence</th>
                      <th className="pb-3 font-medium">Period</th>
                      <th className="pb-3 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyses.map((analysis) => (
                      <tr key={analysis.id} className="border-b border-gray-800">
                        <td className="py-4 text-white font-mono">
                          TIC {analysis.tic_id}
                        </td>
                        <td className="py-4 text-gray-400">
                          {new Date(analysis.created_at).toLocaleDateString()}
                        </td>
                        <td className="py-4">
                          <span className={`px-2 py-1 rounded text-xs ${getStatusBadge(analysis.status)}`}>
                            {analysis.status}
                          </span>
                        </td>
                        <td className="py-4">
                          {analysis.result?.vetting?.disposition ? (
                            <span className={`px-2 py-1 rounded text-xs ${getDispositionBadge(analysis.result.vetting.disposition)}`}>
                              {analysis.result.vetting.disposition.replace(/_/g, ' ')}
                            </span>
                          ) : (
                            <span className="text-gray-500">-</span>
                          )}
                        </td>
                        <td className="py-4 text-white">
                          {analysis.result?.confidence
                            ? `${(analysis.result.confidence * 100).toFixed(1)}%`
                            : '-'}
                        </td>
                        <td className="py-4 text-white">
                          {analysis.result?.period_days
                            ? `${analysis.result.period_days.toFixed(4)}d`
                            : '-'}
                        </td>
                        <td className="py-4">
                          <Link
                            href={`/results/${analysis.id}`}
                            className="text-blue-400 hover:underline text-sm"
                          >
                            View Details
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>

          {/* Quick Tips */}
          <Card className="p-6 mt-6 bg-blue-900/20 border-blue-500/30">
            <h3 className="text-lg font-semibold text-blue-400 mb-2">💡 Tips</h3>
            <ul className="text-gray-300 text-sm space-y-1">
              <li>• Search for known exoplanet hosts to validate results (e.g., TIC 307210830 = TOI-700)</li>
              <li>• Analyses with high confidence ({">"} 85%) and PLANET_CANDIDATE disposition are promising</li>
              <li>• Check all 3 vetting tests pass for best candidates</li>
              <li>• Need more analyses? Upgrade to Professional for unlimited targets</li>
            </ul>
          </Card>
        </div>
      </main>

      <Footer />
    </div>
  );
}
