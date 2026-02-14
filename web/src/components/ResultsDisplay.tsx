'use client'

/**
 * Results Display Component
 *
 * Shows inference results with visualization
 */

import { InferenceResult } from '@/lib/supabase'
import { getModelById } from '@/lib/api-client'
import { Download, Clock, Sparkles } from 'lucide-react'
import { useState } from 'react'

interface ResultsDisplayProps {
  result: InferenceResult
  modelId: string
}

export function ResultsDisplay({ result, modelId }: ResultsDisplayProps) {
  const model = getModelById(modelId)
  const [downloadFormat, setDownloadFormat] = useState<'json' | 'csv'>('json')

  const handleDownload = () => {
    const data = downloadFormat === 'json'
      ? JSON.stringify(result, null, 2)
      : convertToCSV(result)

    const blob = new Blob([data], { type: downloadFormat === 'json' ? 'application/json' : 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `larun-result-${Date.now()}.${downloadFormat}`
    a.click()
    URL.revokeObjectURL(url)
  }

  const convertToCSV = (result: InferenceResult): string => {
    let csv = 'Class,Probability\n'
    Object.entries(result.probabilities).forEach(([cls, prob]) => {
      csv += `${cls},${prob}\n`
    })
    return csv
  }

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-50 border-green-200'
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-50 border-yellow-200'
    return 'text-red-600 bg-red-50 border-red-200'
  }

  return (
    <div className="card space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-2xl mb-2">Analysis Results</h3>
        <p className="text-sm text-larun-medium-gray">
          Model: {model?.name}
        </p>
      </div>

      {/* Classification */}
      <div className="border-2 border-larun-black rounded-lg p-6 bg-larun-lighter-gray">
        <div className="flex items-center gap-3 mb-4">
          <Sparkles className="w-6 h-6 text-larun-black" />
          <h4 className="text-lg font-medium">Classification</h4>
        </div>
        <p className="text-3xl font-medium text-larun-black mb-4">
          {result.classification.replace(/_/g, ' ').toUpperCase()}
        </p>
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border-2 ${getConfidenceColor(result.confidence)}`}>
          <span className="text-sm font-medium">
            {(result.confidence * 100).toFixed(1)}% Confidence
          </span>
        </div>
      </div>

      {/* Probabilities */}
      <div>
        <h4 className="text-lg font-medium mb-4">Class Probabilities</h4>
        <div className="space-y-3">
          {Object.entries(result.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([cls, prob]) => (
              <div key={cls}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-larun-dark-gray capitalize">
                    {cls.replace(/_/g, ' ')}
                  </span>
                  <span className="font-medium text-larun-black">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-larun-lighter-gray rounded-full overflow-hidden">
                  <div
                    className="h-full bg-larun-black transition-all"
                    style={{ width: `${prob * 100}%` }}
                  />
                </div>
              </div>
            ))}
        </div>
      </div>

      {/* Metadata */}
      <div className="grid grid-cols-2 gap-4 pt-4 border-t border-larun-light-gray">
        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-larun-medium-gray" />
          <span className="text-larun-medium-gray">Inference Time:</span>
          <span className="font-medium text-larun-black">
            {result.inference_time_ms.toFixed(1)} ms
          </span>
        </div>
        {result.memory_used_kb && (
          <div className="flex items-center gap-2 text-sm">
            <span className="text-larun-medium-gray">Memory:</span>
            <span className="font-medium text-larun-black">
              {result.memory_used_kb.toFixed(0)} KB
            </span>
          </div>
        )}
      </div>

      {/* Download */}
      <div className="pt-4 border-t border-larun-light-gray">
        <div className="flex items-center gap-3">
          <select
            value={downloadFormat}
            onChange={(e) => setDownloadFormat(e.target.value as 'json' | 'csv')}
            className="input flex-1"
          >
            <option value="json">JSON</option>
            <option value="csv">CSV</option>
          </select>
          <button onClick={handleDownload} className="btn btn-outline">
            <Download className="w-4 h-4" />
            Download
          </button>
        </div>
      </div>
    </div>
  )
}
