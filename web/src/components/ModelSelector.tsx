'use client'

import { TINYML_MODELS, getModelById } from '@/lib/api-client'

const CATEGORY_CONFIG: Record<string, { label: string; badge: string }> = {
  exoplanet: { label: 'Exoplanet Detection', badge: 'bg-blue-50 border-blue-200 text-blue-700' },
  stellar:   { label: 'Stellar Analysis',    badge: 'bg-amber-50 border-amber-200 text-amber-700' },
  transient: { label: 'Transient Events',    badge: 'bg-red-50 border-red-200 text-red-700' },
  galactic:  { label: 'Galactic Science',    badge: 'bg-purple-50 border-purple-200 text-purple-700' },
}

// Preserve display order
const CATEGORY_ORDER = ['exoplanet', 'stellar', 'transient', 'galactic']

interface ModelSelectorProps {
  selectedModel: string
  onModelSelect: (modelId: string) => void
  disabled?: boolean
}

export function ModelSelector({ selectedModel, onModelSelect, disabled }: ModelSelectorProps) {
  const selected = getModelById(selectedModel)

  const grouped = CATEGORY_ORDER.reduce((acc, cat) => {
    acc[cat] = TINYML_MODELS.filter(m => m.category === cat)
    return acc
  }, {} as Record<string, typeof TINYML_MODELS>)

  return (
    <div className={disabled ? 'opacity-50 pointer-events-none select-none' : ''}>
      {/* Card picker grouped by category */}
      <div className="space-y-4">
        {CATEGORY_ORDER.map((cat) => {
          const models = grouped[cat]
          if (!models?.length) return null
          const cfg = CATEGORY_CONFIG[cat]
          return (
            <div key={cat}>
              <p className="text-xs font-semibold text-larun-medium-gray uppercase tracking-wide mb-2">
                {cfg.label}
              </p>
              <div className="space-y-1.5">
                {models.map((model) => {
                  const isSelected = selectedModel === model.id
                  return (
                    <button
                      key={model.id}
                      type="button"
                      onClick={() => onModelSelect(model.id)}
                      className={`w-full text-left px-4 py-3 rounded-lg border-2 transition-all ${
                        isSelected
                          ? 'border-larun-black bg-larun-lighter-gray'
                          : 'border-larun-light-gray hover:border-larun-medium-gray hover:bg-larun-lighter-gray'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <p className="text-sm font-semibold text-larun-black leading-tight">
                            {model.name}
                          </p>
                          <p className="text-xs text-larun-medium-gray mt-0.5 truncate">
                            {model.use_case}
                          </p>
                        </div>
                        <span className={`shrink-0 text-xs px-2 py-0.5 rounded border font-medium ${cfg.badge}`}>
                          {(model.accuracy * 100).toFixed(0)}%
                        </span>
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
          )
        })}
      </div>

      {/* Detail panel for selected model */}
      {selected && (
        <div className="mt-5 p-4 bg-larun-lighter-gray rounded-lg border border-larun-light-gray space-y-3">
          <div>
            <p className="text-xs font-semibold text-larun-medium-gray uppercase tracking-wide mb-1">
              About this model
            </p>
            <p className="text-sm text-larun-dark-gray leading-relaxed">
              {selected.description}
            </p>
          </div>

          <div className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
            <span className="text-larun-medium-gray">
              <span className="font-medium text-larun-black">Data source:</span>{' '}
              {selected.data_source}
            </span>
            <span className="text-larun-medium-gray">
              <span className="font-medium text-larun-black">Model size:</span>{' '}
              {selected.size_kb} KB
            </span>
            <span className="text-larun-medium-gray">
              <span className="font-medium text-larun-black">Input length:</span>{' '}
              {selected.input_length.toLocaleString()} points
            </span>
          </div>

          <div>
            <p className="text-xs font-medium text-larun-black mb-1.5">Output classes:</p>
            <div className="flex flex-wrap gap-1.5">
              {selected.classes.map((cls) => (
                <span
                  key={cls}
                  className="text-xs px-2 py-0.5 bg-white rounded border border-larun-light-gray capitalize"
                >
                  {cls.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
