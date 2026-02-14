'use client'

/**
 * Model Selector Component
 *
 * Dropdown to select from 8 TinyML models
 */

import { TINYML_MODELS, getModelById } from '@/lib/api-client'
import { ChevronDown } from 'lucide-react'

interface ModelSelectorProps {
  selectedModel: string
  onModelSelect: (modelId: string) => void
  disabled?: boolean
}

export function ModelSelector({ selectedModel, onModelSelect, disabled }: ModelSelectorProps) {
  const selected = getModelById(selectedModel)

  return (
    <div>
      <div className="relative">
        <select
          value={selectedModel}
          onChange={(e) => onModelSelect(e.target.value)}
          disabled={disabled}
          className="input appearance-none pr-10 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {TINYML_MODELS.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} ({model.accuracy * 100}% accurate)
            </option>
          ))}
        </select>
        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-larun-medium-gray pointer-events-none" />
      </div>

      {selected && (
        <div className="mt-4 p-4 bg-larun-lighter-gray rounded-lg">
          <h4 className="text-sm font-medium text-larun-black mb-2">
            {selected.name}
          </h4>
          <p className="text-sm text-larun-medium-gray mb-3">
            {selected.description}
          </p>
          <div className="flex gap-4 text-xs text-larun-medium-gray">
            <div>
              <span className="font-medium text-larun-black">Accuracy:</span>{' '}
              {(selected.accuracy * 100).toFixed(1)}%
            </div>
            <div>
              <span className="font-medium text-larun-black">Size:</span>{' '}
              {selected.size_kb} KB
            </div>
            <div>
              <span className="font-medium text-larun-black">Input:</span>{' '}
              {selected.input_length} points
            </div>
          </div>

          <div className="mt-3">
            <p className="text-xs font-medium text-larun-black mb-2">Output Classes:</p>
            <div className="flex flex-wrap gap-2">
              {selected.classes.map((cls) => (
                <span
                  key={cls}
                  className="text-xs px-2 py-1 bg-larun-white rounded border border-larun-light-gray"
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
