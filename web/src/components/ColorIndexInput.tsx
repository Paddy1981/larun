'use client'

import { useState } from 'react'
import { ChevronDown, ChevronUp, Info } from 'lucide-react'

export interface ColorIndices {
  bv:    string
  vr:    string
  bp_rp: string
  jh:    string
  hk:    string
}

interface Field {
  key:   keyof ColorIndices
  label: string
  latex: string        // display symbol
  range: string
  hint:  string
}

const FIELD_GROUPS: { group: string; source: string; fields: Field[] }[] = [
  {
    group: 'Johnson–Cousins',
    source: 'Standard broadband photometry — most catalogues (SIMBAD, VizieR)',
    fields: [
      { key: 'bv',  label: 'B−V',  latex: 'B−V',  range: '−0.4 to 1.8', hint: 'Johnson B minus V magnitude.  Primary spectral indicator.' },
      { key: 'vr',  label: 'V−R',  latex: 'V−R',  range: '−0.1 to 0.9', hint: 'Johnson V minus Cousins R magnitude.' },
    ],
  },
  {
    group: 'Gaia DR3',
    source: 'ESA Gaia archive — gaia.esac.esa.int or VizieR I/355',
    fields: [
      { key: 'bp_rp', label: 'BP−RP', latex: 'BP−RP', range: '−0.5 to 3.5', hint: 'Gaia blue (BP) minus red (RP) photometer. Very precise for >100 million stars.' },
    ],
  },
  {
    group: '2MASS Near-Infrared',
    source: 'IRSA / VizieR II/246 — 2MASS All-Sky Point Source Catalog',
    fields: [
      { key: 'jh', label: 'J−H', latex: 'J−H', range: '0.0 to 0.8',  hint: '2MASS J (1.24 µm) minus H (1.66 µm). Sensitive to cool stars.' },
      { key: 'hk', label: 'H−K', latex: 'H−K', range: '0.0 to 0.4',  hint: '2MASS H (1.66 µm) minus Ks (2.16 µm). Sensitive to circumstellar emission.' },
    ],
  },
]

// Typical B−V per spectral type as a quick reference
const REFERENCE: { type: string; bv: string; bp_rp: string; teff: string }[] = [
  { type: 'O', bv: '< −0.25', bp_rp: '< −0.15', teff: '> 30 000 K' },
  { type: 'B', bv: '−0.25 to 0.00', bp_rp: '−0.15 to 0.10', teff: '10 000 – 30 000 K' },
  { type: 'A', bv: '0.00 to 0.20', bp_rp: '0.10 to 0.50', teff: '7 500 – 10 000 K' },
  { type: 'F', bv: '0.20 to 0.48', bp_rp: '0.50 to 0.85', teff: '6 000 – 7 500 K' },
  { type: 'G', bv: '0.48 to 0.68', bp_rp: '0.85 to 1.35', teff: '5 200 – 6 000 K' },
  { type: 'K', bv: '0.68 to 1.18', bp_rp: '1.35 to 2.20', teff: '3 700 – 5 200 K' },
  { type: 'M', bv: '> 1.18',       bp_rp: '> 2.20',       teff: '< 3 700 K' },
]

interface ColorIndexInputProps {
  values: ColorIndices
  onChange: (values: ColorIndices) => void
  disabled?: boolean
}

export function ColorIndexInput({ values, onChange, disabled }: ColorIndexInputProps) {
  const [showRef, setShowRef] = useState(false)
  const [tooltip, setTooltip] = useState<string | null>(null)

  const set = (key: keyof ColorIndices, val: string) =>
    onChange({ ...values, [key]: val })

  const hasAtLeastOne = Object.values(values).some(v => v.trim() !== '' && !isNaN(Number(v)))

  return (
    <div className={disabled ? 'opacity-50 pointer-events-none' : ''}>
      <p className="text-sm text-larun-medium-gray mb-5">
        Enter photometric colour indices from any combination of sources below.{' '}
        <span className="font-medium text-larun-black">At least one value is required.</span>
      </p>

      {/* Input groups */}
      <div className="space-y-6">
        {FIELD_GROUPS.map(({ group, source, fields }) => (
          <div key={group}>
            <p className="text-xs font-semibold text-larun-medium-gray uppercase tracking-wide mb-0.5">
              {group}
            </p>
            <p className="text-xs text-larun-medium-gray mb-3">{source}</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {fields.map(({ key, label, range, hint }) => (
                <div key={key} className="relative">
                  <label className="block text-sm font-medium text-larun-black mb-1">
                    <span className="font-mono">{label}</span>
                    <span className="text-larun-medium-gray font-normal ml-2 text-xs">optional</span>
                    <button
                      type="button"
                      className="ml-1 text-larun-medium-gray hover:text-larun-black"
                      onMouseEnter={() => setTooltip(key)}
                      onMouseLeave={() => setTooltip(null)}
                      onFocus={() => setTooltip(key)}
                      onBlur={() => setTooltip(null)}
                    >
                      <Info className="w-3.5 h-3.5 inline" />
                    </button>
                    {tooltip === key && (
                      <span className="absolute z-10 left-0 top-full mt-1 w-64 bg-larun-black text-white text-xs rounded-lg p-2 shadow-lg pointer-events-none">
                        {hint}
                      </span>
                    )}
                  </label>
                  <div className="relative">
                    <input
                      type="number"
                      step="0.01"
                      placeholder={`e.g. ${range.split(' ')[0]}`}
                      value={values[key]}
                      onChange={(e) => set(key, e.target.value)}
                      className="input pr-20 font-mono text-sm"
                    />
                    <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-larun-medium-gray pointer-events-none whitespace-nowrap">
                      {range}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Validation hint */}
      {!hasAtLeastOne && (
        <p className="mt-4 text-xs text-amber-600">
          Enter at least one colour index to enable analysis.
        </p>
      )}

      {/* Reference table toggle */}
      <button
        type="button"
        onClick={() => setShowRef(v => !v)}
        className="mt-5 flex items-center gap-1.5 text-xs text-larun-medium-gray hover:text-larun-black transition-colors"
      >
        {showRef ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        {showRef ? 'Hide' : 'Show'} typical values per spectral type
      </button>

      {showRef && (
        <div className="mt-3 overflow-x-auto rounded-lg border border-larun-light-gray">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-larun-lighter-gray border-b border-larun-light-gray text-left">
                <th className="px-3 py-2 font-semibold text-larun-black">Type</th>
                <th className="px-3 py-2 font-semibold text-larun-black font-mono">B−V</th>
                <th className="px-3 py-2 font-semibold text-larun-black font-mono">BP−RP</th>
                <th className="px-3 py-2 font-semibold text-larun-black">T<sub>eff</sub></th>
              </tr>
            </thead>
            <tbody>
              {REFERENCE.map((row) => (
                <tr key={row.type} className="border-b border-larun-lighter-gray last:border-0">
                  <td className="px-3 py-2 font-bold text-larun-black">{row.type}</td>
                  <td className="px-3 py-2 font-mono text-larun-dark-gray">{row.bv}</td>
                  <td className="px-3 py-2 font-mono text-larun-dark-gray">{row.bp_rp}</td>
                  <td className="px-3 py-2 text-larun-medium-gray">{row.teff}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
