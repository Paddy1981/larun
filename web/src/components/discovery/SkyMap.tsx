'use client'

/**
 * SkyMap — Aladin Lite integration for sky region selection
 * Loads Aladin Lite v3 from CDN; calls onSelect when user picks a region.
 */

import { useEffect, useRef, useState } from 'react'
import { MapPin, Target, Loader2 } from 'lucide-react'

interface SkyMapProps {
  onSelect?: (ra: number, dec: number) => void
  initialRA?: number
  initialDec?: number
  radiusDeg?: number
  height?: number
}

declare global {
  interface Window {
    A?: {
      init: (container: string | HTMLElement, options: Record<string, unknown>) => Promise<unknown>
      aladin: (container: string | HTMLElement, options: Record<string, unknown>) => AladinInstance
    }
  }
}

interface AladinInstance {
  getRaDec: () => [number, number]
  gotoRaDec: (ra: number, dec: number) => void
  setFov: (fov: number) => void
  on: (event: string, callback: (...args: unknown[]) => void) => void
}

const POPULAR_TARGETS: { name: string; ra: number; dec: number; description: string }[] = [
  { name: 'Pleiades',       ra: 56.75,  dec: 24.12,   description: 'Open cluster, M45' },
  { name: 'Galactic Bulge', ra: 266.42, dec: -29.01,  description: 'OGLE microlensing field' },
  { name: 'Kepler Field',   ra: 291.0,  dec: 44.5,    description: 'Original Kepler FoV' },
  { name: 'TESS CVZ South', ra: 90.0,   dec: -66.0,   description: 'TESS continuous viewing zone' },
  { name: 'LMC',            ra: 80.89,  dec: -69.76,  description: 'Large Magellanic Cloud' },
  { name: 'Orion OB1',      ra: 83.82,  dec: -5.39,   description: 'Active star-forming region' },
]

export function SkyMap({ onSelect, initialRA = 56.75, initialDec = 24.12, radiusDeg = 1.0, height = 400 }: SkyMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const aladinRef = useRef<AladinInstance | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [currentCoords, setCurrentCoords] = useState({ ra: initialRA, dec: initialDec })
  const [manualRA, setManualRA] = useState(String(initialRA))
  const [manualDec, setManualDec] = useState(String(initialDec))

  useEffect(() => {
    // Timeout — show fallback if Aladin doesn't load within 8 seconds
    const timeout = setTimeout(() => {
      if (!loaded) setLoaded(false) // triggers fallback UI
    }, 8000)

    // Load Aladin Lite v3 script
    const scriptId = 'aladin-lite-script'
    if (document.getElementById(scriptId)) {
      initAladin()
      clearTimeout(timeout)
      return
    }
    const script = document.createElement('script')
    script.id = scriptId
    script.src = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js'
    script.charset = 'utf-8'
    script.onload = () => { clearTimeout(timeout); initAladin() }
    script.onerror = () => clearTimeout(timeout)
    document.head.appendChild(script)

    return () => clearTimeout(timeout)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  function initAladin() {
    if (!containerRef.current || !window.A) return
    try {
      const aladin = window.A.aladin(containerRef.current, {
        survey: 'P/DSS2/color',
        fov: radiusDeg * 3,
        target: `${initialRA} ${initialDec}`,
        showControl: false,
        showFrame: false,
        showCooGrid: true,
        showSimbadPointerControl: false,
        showShareControl: false,
        showFullscreenControl: false,
        cooFrame: 'ICRS',
      })
      aladinRef.current = aladin

      aladin.on('positionChanged', () => {
        const [ra, dec] = aladin.getRaDec()
        setCurrentCoords({ ra: +ra.toFixed(4), dec: +dec.toFixed(4) })
      })

      setLoaded(true)
    } catch {
      // Aladin Lite may fail in SSR/test environments — graceful fallback
      setLoaded(false)
    }
  }

  function handleSelectCurrent() {
    const { ra, dec } = currentCoords
    setManualRA(String(ra))
    setManualDec(String(dec))
    onSelect?.(ra, dec)
  }

  function handleManualGo() {
    const ra = parseFloat(manualRA)
    const dec = parseFloat(manualDec)
    if (isNaN(ra) || isNaN(dec)) return
    if (ra < 0 || ra >= 360 || dec < -90 || dec > 90) return
    aladinRef.current?.gotoRaDec(ra, dec)
    setCurrentCoords({ ra, dec })
    onSelect?.(ra, dec)
  }

  function handlePopularTarget(target: typeof POPULAR_TARGETS[number]) {
    aladinRef.current?.gotoRaDec(target.ra, target.dec)
    setCurrentCoords({ ra: target.ra, dec: target.dec })
    setManualRA(String(target.ra))
    setManualDec(String(target.dec))
    onSelect?.(target.ra, target.dec)
  }

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-[#e5e7eb] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-[#1a73e8]" />
          <span className="font-medium text-[#202124] text-sm">Sky Region Selection</span>
        </div>
        <span className="text-xs text-[#5f6368]">
          RA {currentCoords.ra.toFixed(3)}° / Dec {currentCoords.dec > 0 ? '+' : ''}{currentCoords.dec.toFixed(3)}°
        </span>
      </div>

      {/* Map container */}
      <div className="relative" style={{ height }}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

        {!loaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#f8f9fa] gap-3">
            <Loader2 className="w-6 h-6 text-[#1a73e8] animate-spin" />
            <span className="text-sm text-[#5f6368]">Loading sky map…</span>
            <span className="text-xs text-[#9ca3af]">Use the coordinate inputs below while it loads</span>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-[#e5e7eb] space-y-3">
        {/* Manual coords */}
        <div className="flex gap-2 items-center">
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">RA (deg)</label>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualRA}
              onChange={e => setManualRA(e.target.value)}
              placeholder="0–360"
            />
          </div>
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">Dec (deg)</label>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualDec}
              onChange={e => setManualDec(e.target.value)}
              placeholder="-90 to +90"
            />
          </div>
          <div className="pt-5">
            <button
              onClick={handleManualGo}
              className="btn btn-primary text-sm px-4 py-2"
            >
              Go
            </button>
          </div>
        </div>

        {/* Select current center */}
        <button
          onClick={handleSelectCurrent}
          className="w-full flex items-center justify-center gap-2 btn btn-outline text-sm py-2"
        >
          <MapPin className="w-4 h-4" />
          Select current center
        </button>

        {/* Popular targets */}
        <div>
          <p className="text-xs text-[#5f6368] mb-2">Popular targets:</p>
          <div className="flex flex-wrap gap-2">
            {POPULAR_TARGETS.map(t => (
              <button
                key={t.name}
                onClick={() => handlePopularTarget(t)}
                title={t.description}
                className="text-xs px-2 py-1 rounded-full bg-[#f1f3f4] text-[#202124] hover:bg-[#e8eaed] transition-colors"
              >
                {t.name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
