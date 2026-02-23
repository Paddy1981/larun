'use client'

import { TINYML_MODELS, getModelById } from '@/lib/api-client'

const CATEGORY_CONFIG: Record<string, { label: string; accent: string; badgeSelected: string; badgeUnselected: string }> = {
  exoplanet: { label: 'Exoplanet Detection', accent: '#3b82f6', badgeSelected: 'bg-blue-400/20 text-blue-200',  badgeUnselected: 'bg-blue-50 border border-blue-200 text-blue-700' },
  stellar:   { label: 'Stellar Analysis',    accent: '#f59e0b', badgeSelected: 'bg-amber-400/20 text-amber-200', badgeUnselected: 'bg-amber-50 border border-amber-200 text-amber-700' },
  transient: { label: 'Transient Events',    accent: '#ef4444', badgeSelected: 'bg-red-400/20 text-red-200',    badgeUnselected: 'bg-red-50 border border-red-200 text-red-700' },
  galactic:  { label: 'Galactic Science',    accent: '#a855f7', badgeSelected: 'bg-purple-400/20 text-purple-200', badgeUnselected: 'bg-purple-50 border border-purple-200 text-purple-700' },
}

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
      <div className="space-y-4">
        {CATEGORY_ORDER.map((cat) => {
          const models = grouped[cat]
          if (!models?.length) return null
          const cfg = CATEGORY_CONFIG[cat]
          return (
            <div key={cat}>
              <p className="text-xs font-semibold text-[#9ca3af] uppercase tracking-wide mb-2">
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
                      className={`w-full text-left px-4 py-3 rounded-xl border-2 transition-all duration-150 ${
                        isSelected
                          ? 'bg-[#202124] border-[#202124] shadow-md'
                          : 'bg-white border-[#e5e7eb] hover:border-[#9ca3af] hover:bg-[#f9fafb]'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        {/* Radio indicator */}
                        <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center shrink-0 transition-all ${
                          isSelected
                            ? 'border-white bg-white'
                            : 'border-[#d1d5db] bg-white'
                        }`}>
                          {isSelected && (
                            <div className="w-2 h-2 rounded-full bg-[#202124]" />
                          )}
                        </div>

                        {/* Text */}
                        <div className="min-w-0 flex-1">
                          <p className={`text-sm font-semibold leading-tight ${isSelected ? 'text-white' : 'text-[#202124]'}`}>
                            {model.name}
                          </p>
                          <p className={`text-xs mt-0.5 truncate ${isSelected ? 'text-[#9ca3af]' : 'text-[#6b7280]'}`}>
                            {model.use_case}
                          </p>
                        </div>

                        {/* Accuracy badge */}
                        <span className={`shrink-0 text-xs px-2 py-0.5 rounded-full font-semibold ${
                          isSelected ? cfg.badgeSelected : cfg.badgeUnselected
                        }`}>
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

      {/* Detail panel */}
      {selected && (
        <div className="mt-4 p-4 bg-[#f8f9fa] rounded-xl border border-[#e5e7eb] space-y-3">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-1.5 h-1.5 rounded-full bg-[#202124]" />
            <p className="text-xs font-semibold text-[#374151] uppercase tracking-wide">
              {selected.name}
            </p>
          </div>
          <p className="text-sm text-[#6b7280] leading-relaxed">
            {selected.description}
          </p>

          <div className="grid grid-cols-2 gap-2 pt-1">
            <div className="bg-white rounded-lg p-2.5 border border-[#e5e7eb]">
              <p className="text-xs text-[#9ca3af] mb-0.5">Data source</p>
              <p className="text-xs font-medium text-[#374151]">{selected.data_source}</p>
            </div>
            <div className="bg-white rounded-lg p-2.5 border border-[#e5e7eb]">
              <p className="text-xs text-[#9ca3af] mb-0.5">Model size</p>
              <p className="text-xs font-medium text-[#374151]">{selected.size_kb} KB</p>
            </div>
          </div>

          <div>
            <p className="text-xs text-[#9ca3af] mb-1.5">Output classes</p>
            <div className="flex flex-wrap gap-1.5">
              {selected.classes.map((cls) => (
                <span
                  key={cls}
                  className="text-xs px-2 py-0.5 bg-white rounded-full border border-[#e5e7eb] text-[#374151] capitalize"
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
