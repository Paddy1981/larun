'use client'

/**
 * SkyMap — Canvas-based interactive sky chart (no external CDN)
 * Renders a procedural star field with click-to-select coordinates.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { MapPin, Target, ZoomIn, ZoomOut } from 'lucide-react'

interface SkyMapProps {
  onSelect?: (ra: number, dec: number) => void
  initialRA?: number
  initialDec?: number
  radiusDeg?: number
  height?: number
}

// Popular sky targets
const POPULAR_TARGETS: { name: string; ra: number; dec: number; description: string }[] = [
  { name: 'Pleiades',       ra: 56.75,  dec: 24.12,  description: 'Open cluster, M45' },
  { name: 'Galactic Bulge', ra: 266.42, dec: -29.01, description: 'OGLE microlensing field' },
  { name: 'Kepler Field',   ra: 291.0,  dec: 44.5,   description: 'Original Kepler FoV' },
  { name: 'TESS CVZ South', ra: 90.0,   dec: -66.0,  description: 'TESS continuous viewing zone' },
  { name: 'LMC',            ra: 80.89,  dec: -69.76, description: 'Large Magellanic Cloud' },
  { name: 'Orion OB1',      ra: 83.82,  dec: -5.39,  description: 'Active star-forming region' },
]

// Deterministic star positions from a seed (LCG PRNG)
function mkStars(seed: number, count: number) {
  let s = seed | 0
  const rng = () => { s = (Math.imul(1664525, s) + 1013904223) | 0; return (s >>> 0) / 4294967296 }
  return Array.from({ length: count }, () => ({
    x: rng(), y: rng(),
    mag: rng() * rng() * rng(), // skewed toward dim
    r: rng() < 0.04,            // rare red giant
  }))
}

// Milky Way band: bright blob near galactic plane
function isNearGalacticPlane(ra: number, dec: number): number {
  // Rough galactic plane in equatorial coords (simplified)
  const galLat = Math.abs(dec - 10 * Math.cos((ra - 180) * Math.PI / 180) - 5)
  return Math.max(0, 1 - galLat / 40)
}

export function SkyMap({ onSelect, initialRA = 56.75, initialDec = 24.12, radiusDeg = 1.0, height = 400 }: SkyMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [centerRA, setCenterRA]   = useState(initialRA)
  const [centerDec, setCenterDec] = useState(initialDec)
  const [fovDeg, setFovDeg]       = useState(Math.max(2, radiusDeg * 6))
  const [manualRA, setManualRA]   = useState(String(initialRA))
  const [manualDec, setManualDec] = useState(String(initialDec))
  const [hovering, setHovering]   = useState(false)
  const starsRef = useRef(mkStars(0x1a73e8, 2200))

  // Convert RA/Dec offset from center → canvas px
  const project = useCallback((ra: number, dec: number, w: number, h: number) => {
    const scale = Math.min(w, h) / fovDeg
    const dx = ((ra - centerRA + 540) % 360 - 180) * Math.cos(centerDec * Math.PI / 180) * scale
    const dy = (dec - centerDec) * scale
    return { x: w / 2 - dx, y: h / 2 - dy }
  }, [centerRA, centerDec, fovDeg])

  // Convert canvas px → RA/Dec
  const unproject = useCallback((px: number, py: number, w: number, h: number) => {
    const scale = Math.min(w, h) / fovDeg
    const dx = (w / 2 - px) / scale / Math.cos(centerDec * Math.PI / 180)
    const dy = (h / 2 - py) / scale
    const ra  = ((centerRA + dx) % 360 + 360) % 360
    const dec = Math.max(-90, Math.min(90, centerDec + dy))
    return { ra, dec }
  }, [centerRA, centerDec, fovDeg])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const W = canvas.width, H = canvas.height

    // ── Background ──────────────────────────────────────────────────────────
    const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.7)
    const galBlend = isNearGalacticPlane(centerRA, centerDec)
    const midColor = galBlend > 0.3 ? `rgba(30,35,60,${0.4 + galBlend * 0.3})` : 'rgba(12,18,38,0)'
    bg.addColorStop(0, 'rgba(12,18,38,1)')
    bg.addColorStop(0.5, midColor)
    bg.addColorStop(1, 'rgba(6,9,20,1)')
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, W, H)

    // ── Subtle Milky Way glow ────────────────────────────────────────────────
    if (galBlend > 0.1) {
      const mw = ctx.createLinearGradient(0, 0, W, H)
      mw.addColorStop(0, `rgba(100,120,180,0)`)
      mw.addColorStop(0.5, `rgba(100,120,180,${galBlend * 0.06})`)
      mw.addColorStop(1, `rgba(100,120,180,0)`)
      ctx.fillStyle = mw
      ctx.fillRect(0, 0, W, H)
    }

    // ── RA/Dec grid ─────────────────────────────────────────────────────────
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 0.5
    const gridStep = fovDeg > 10 ? 10 : fovDeg > 3 ? 5 : 1
    for (let ra = 0; ra < 360; ra += gridStep) {
      ctx.beginPath()
      for (let d = -90; d <= 90; d += 2) {
        const p = project(ra, d, W, H)
        d === -90 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
    }
    for (let dec = -80; dec <= 80; dec += gridStep) {
      ctx.beginPath()
      for (let ra = 0; ra <= 360; ra += 3) {
        const p = project(ra, dec, W, H)
        ra === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
    }

    // ── Stars ────────────────────────────────────────────────────────────────
    const scale = Math.min(W, H) / fovDeg
    starsRef.current.forEach(star => {
      const starRA  = star.x * 360
      const starDec = star.y * 180 - 90
      const p = project(starRA, starDec, W, H)
      if (p.x < -4 || p.x > W + 4 || p.y < -4 || p.y > H + 4) return

      const brightness = 0.2 + star.mag * 0.8
      const radius = Math.max(0.3, (1 - star.mag) * (scale / 120) + 0.4)

      if (radius < 0.3) return // too small to draw

      // Glow for bright stars
      if (star.mag < 0.05) {
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, radius * 5)
        const color = star.r ? '255,180,100' : '200,210,255'
        glow.addColorStop(0, `rgba(${color},${brightness * 0.4})`)
        glow.addColorStop(1, 'rgba(0,0,0,0)')
        ctx.fillStyle = glow
        ctx.beginPath()
        ctx.arc(p.x, p.y, radius * 5, 0, Math.PI * 2)
        ctx.fill()
      }

      const color = star.r
        ? `rgba(255,160,80,${brightness})`
        : star.mag < 0.1
        ? `rgba(200,220,255,${brightness})`
        : `rgba(220,230,255,${brightness * 0.9})`
      ctx.fillStyle = color
      ctx.beginPath()
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2)
      ctx.fill()
    })

    // ── Popular target markers ───────────────────────────────────────────────
    ctx.font = `${Math.max(9, scale / 15)}px Inter, sans-serif`
    POPULAR_TARGETS.forEach(t => {
      const p = project(t.ra, t.dec, W, H)
      if (p.x < -20 || p.x > W + 20 || p.y < -20 || p.y > H + 20) return
      // Crosshair dot
      ctx.strokeStyle = 'rgba(255,200,80,0.7)'
      ctx.lineWidth = 1
      ctx.beginPath()
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2)
      ctx.stroke()
      // Label
      ctx.fillStyle = 'rgba(255,200,80,0.85)'
      ctx.fillText(t.name, p.x + 8, p.y - 4)
    })

    // ── Selected region circle ───────────────────────────────────────────────
    const selPx = project(centerRA, centerDec, W, H)
    const radiusPx = (radiusDeg / fovDeg) * Math.min(W, H)
    // Outer glow
    const glow = ctx.createRadialGradient(selPx.x, selPx.y, radiusPx * 0.85, selPx.x, selPx.y, radiusPx * 1.15)
    glow.addColorStop(0, 'rgba(26,115,232,0.12)')
    glow.addColorStop(0.5, 'rgba(26,115,232,0.06)')
    glow.addColorStop(1, 'rgba(26,115,232,0)')
    ctx.fillStyle = glow
    ctx.beginPath()
    ctx.arc(selPx.x, selPx.y, radiusPx * 1.15, 0, Math.PI * 2)
    ctx.fill()
    // Ring
    ctx.strokeStyle = 'rgba(26,115,232,0.85)'
    ctx.lineWidth = 1.5
    ctx.setLineDash([4, 3])
    ctx.beginPath()
    ctx.arc(selPx.x, selPx.y, radiusPx, 0, Math.PI * 2)
    ctx.stroke()
    ctx.setLineDash([])
    // Crosshair
    ctx.strokeStyle = 'rgba(26,115,232,0.6)'
    ctx.lineWidth = 1
    const cs = 10
    ctx.beginPath()
    ctx.moveTo(selPx.x - cs, selPx.y); ctx.lineTo(selPx.x + cs, selPx.y)
    ctx.moveTo(selPx.x, selPx.y - cs); ctx.lineTo(selPx.x, selPx.y + cs)
    ctx.stroke()
  }, [centerRA, centerDec, fovDeg, radiusDeg, project])

  // Redraw on state change
  useEffect(() => {
    draw()
  }, [draw])

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const obs = new ResizeObserver(() => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = rect.width * window.devicePixelRatio
      canvas.height = rect.height * window.devicePixelRatio
      canvas.style.width = `${rect.width}px`
      canvas.style.height = `${rect.height}px`
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      draw()
    })
    obs.observe(canvas)
    return () => obs.disconnect()
  }, [draw])

  // Click to select
  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const { ra, dec } = unproject(px, py, rect.width, rect.height)
    const raR = +ra.toFixed(4), decR = +dec.toFixed(4)
    setCenterRA(raR)
    setCenterDec(decR)
    setManualRA(String(raR))
    setManualDec(String(decR))
    onSelect?.(raR, decR)
  }, [unproject, onSelect])

  function handleManualGo() {
    const ra = parseFloat(manualRA), dec = parseFloat(manualDec)
    if (isNaN(ra) || isNaN(dec) || ra < 0 || ra >= 360 || dec < -90 || dec > 90) return
    setCenterRA(ra); setCenterDec(dec); onSelect?.(ra, dec)
  }

  function handlePopularTarget(t: typeof POPULAR_TARGETS[number]) {
    setCenterRA(t.ra); setCenterDec(t.dec)
    setManualRA(String(t.ra)); setManualDec(String(t.dec))
    onSelect?.(t.ra, t.dec)
  }

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#e5e7eb] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Target className="w-4 h-4 text-[#1a73e8]" />
          <span className="font-medium text-[#202124] text-sm">Sky Region Selection</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs font-mono text-[#5f6368]">
            {centerRA.toFixed(3)}° / {centerDec > 0 ? '+' : ''}{centerDec.toFixed(3)}°
          </span>
          {/* Zoom */}
          <div className="flex items-center gap-1">
            <button onClick={() => setFovDeg(f => Math.max(0.5, f / 1.6))}
              className="w-6 h-6 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => setFovDeg(f => Math.min(90, f * 1.6))}
              className="w-6 h-6 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Canvas */}
      <div
        className="relative bg-[#060914] cursor-crosshair"
        style={{ height }}
        onMouseEnter={() => setHovering(true)}
        onMouseLeave={() => setHovering(false)}
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />
        {/* Click hint */}
        {hovering && (
          <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-xs text-white/50 pointer-events-none select-none">
            Click to select · Scroll buttons to zoom
          </div>
        )}
        {/* FOV badge */}
        <div className="absolute top-2 right-2 text-[10px] font-mono text-white/40 select-none">
          {fovDeg.toFixed(1)}° FoV
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-[#e5e7eb] space-y-3">
        {/* Manual coords */}
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">RA (deg)</label>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualRA}
              onChange={e => setManualRA(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()}
              placeholder="0–360"
            />
          </div>
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">Dec (deg)</label>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualDec}
              onChange={e => setManualDec(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()}
              placeholder="-90 to +90"
            />
          </div>
          <button onClick={handleManualGo} className="btn btn-primary text-sm px-4 py-2">Go</button>
        </div>

        {/* Select current */}
        <button
          onClick={() => { onSelect?.(centerRA, centerDec) }}
          className="w-full flex items-center justify-center gap-2 btn btn-outline text-sm py-2"
        >
          <MapPin className="w-4 h-4" />
          Use this region (RA {centerRA.toFixed(2)}°, Dec {centerDec.toFixed(2)}°)
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
                className="text-xs px-2.5 py-1 rounded-full bg-[#f1f3f4] text-[#202124] hover:bg-[#e8eaed] transition-colors"
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
