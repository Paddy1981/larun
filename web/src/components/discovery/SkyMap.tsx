'use client'

/**
 * SkyMap — Canvas-based interactive sky chart
 * Screen-space star generation: always dense, loads instantly, no CDN.
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

const POPULAR_TARGETS = [
  { name: 'Pleiades',       ra: 56.75,  dec: 24.12,  description: 'Open cluster, M45' },
  { name: 'Galactic Bulge', ra: 266.42, dec: -29.01, description: 'OGLE microlensing field' },
  { name: 'Kepler Field',   ra: 291.0,  dec: 44.5,   description: 'Original Kepler FoV' },
  { name: 'TESS CVZ South', ra: 90.0,   dec: -66.0,  description: 'TESS continuous viewing zone' },
  { name: 'LMC',            ra: 80.89,  dec: -69.76, description: 'Large Magellanic Cloud' },
  { name: 'Orion OB1',      ra: 83.82,  dec: -5.39,  description: 'Active star-forming region' },
]

// Fast xorshift32 RNG
function rng32(seed: number) {
  let s = (seed ^ 0x5f3759df) | 1
  return () => {
    s ^= s << 13
    s ^= s >> 17
    s ^= s << 5
    return (s >>> 0) / 4294967296
  }
}

interface Star { x: number; y: number; r: number; color: string; alpha: number }

// Generate screen-space stars. Seed is quantised so panning nearby reuses same field.
function genStars(seedRA: number, seedDec: number, W: number, H: number): Star[] {
  const qRA  = Math.round(seedRA  / 8) * 8
  const qDec = Math.round(seedDec / 8) * 8
  const seed = (qRA * 1301 + qDec * 4507 + 0xdeadbeef) | 0
  const rand = rng32(seed)

  const stars: Star[] = []

  // Dim background stars (many, tiny)
  for (let i = 0; i < 520; i++) {
    const mag  = rand() * rand()            // skew toward dim
    const r    = 0.35 + (1 - mag) * 0.55
    const a    = 0.15 + mag * 0.5
    const type = rand()
    const color = type < 0.06 ? '#ffa060'  // red giants
                : type < 0.12 ? '#a0c0ff'  // blue-white
                :               '#d8e8ff'  // white/yellow-white
    stars.push({ x: rand() * W, y: rand() * H, r, color, alpha: a })
  }

  // Medium stars
  for (let i = 0; i < 160; i++) {
    const mag  = rand() * rand() * rand()
    const r    = 0.7 + (1 - mag) * 1.1
    const a    = 0.45 + mag * 0.45
    const type = rand()
    const color = type < 0.05 ? '#ff8844'
                : type < 0.10 ? '#88aaff'
                :               '#e8f0ff'
    stars.push({ x: rand() * W, y: rand() * H, r, color, alpha: a })
  }

  // Bright stars (few, with diffraction glow)
  for (let i = 0; i < 28; i++) {
    const r    = 1.4 + rand() * 1.6
    const type = rand()
    const color = type < 0.15 ? '#ff9955' : type < 0.25 ? '#99bbff' : '#ffffff'
    stars.push({ x: rand() * W, y: rand() * H, r, color, alpha: 0.92 + rand() * 0.08 })
  }

  return stars
}

export function SkyMap({
  onSelect, initialRA = 56.75, initialDec = 24.12, radiusDeg = 1.0, height = 400,
}: SkyMapProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const starsRef     = useRef<Star[]>([])
  const sizeRef      = useRef({ w: 0, h: 0 })

  const [centerRA,  setCenterRA]  = useState(initialRA)
  const [centerDec, setCenterDec] = useState(initialDec)
  const [fovDeg,    setFovDeg]    = useState(Math.max(3, radiusDeg * 5))
  const [manualRA,  setManualRA]  = useState(String(initialRA))
  const [manualDec, setManualDec] = useState(String(initialDec))

  // ── coordinate helpers ────────────────────────────────────────────────────

  const project = useCallback((ra: number, dec: number, W: number, H: number) => {
    const scale = Math.min(W, H) / fovDeg
    const dx = ((ra - centerRA + 540) % 360 - 180) * Math.cos(centerDec * Math.PI / 180) * scale
    const dy = (dec - centerDec) * scale
    return { x: W / 2 - dx, y: H / 2 - dy }
  }, [centerRA, centerDec, fovDeg])

  const unproject = useCallback((px: number, py: number, W: number, H: number) => {
    const scale = Math.min(W, H) / fovDeg
    const dx = (W / 2 - px) / scale / Math.cos(centerDec * Math.PI / 180)
    const dy = (H / 2 - py) / scale
    return {
      ra:  ((centerRA + dx) % 360 + 360) % 360,
      dec: Math.max(-90, Math.min(90, centerDec + dy)),
    }
  }, [centerRA, centerDec, fovDeg])

  // ── draw ──────────────────────────────────────────────────────────────────

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const { w: W, h: H } = sizeRef.current
    if (W === 0 || H === 0) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Background gradient
    const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75)
    bg.addColorStop(0,   '#0d1526')
    bg.addColorStop(0.6, '#080e1c')
    bg.addColorStop(1,   '#04090f')
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, W, H)

    // Milky Way glow — elliptical band
    const galIntensity = Math.max(0, 1 - Math.abs(centerDec + 5 * Math.sin(centerRA * 0.0174)) / 35)
    if (galIntensity > 0.05) {
      const mw = ctx.createLinearGradient(0, H * 0.1, W, H * 0.9)
      mw.addColorStop(0,   'rgba(60,80,140,0)')
      mw.addColorStop(0.35, `rgba(60,80,140,${galIntensity * 0.07})`)
      mw.addColorStop(0.5,  `rgba(80,100,160,${galIntensity * 0.12})`)
      mw.addColorStop(0.65, `rgba(60,80,140,${galIntensity * 0.07})`)
      mw.addColorStop(1,   'rgba(60,80,140,0)')
      ctx.fillStyle = mw
      ctx.fillRect(0, 0, W, H)
      // Subtle dust lane
      const dust = ctx.createLinearGradient(0, H * 0.35, W, H * 0.65)
      dust.addColorStop(0,   'rgba(0,0,0,0)')
      dust.addColorStop(0.5, `rgba(0,0,0,${galIntensity * 0.04})`)
      dust.addColorStop(1,   'rgba(0,0,0,0)')
      ctx.fillStyle = dust
      ctx.fillRect(0, 0, W, H)
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.055)'
    ctx.lineWidth = 0.5
    const gridStep = fovDeg > 15 ? 15 : fovDeg > 5 ? 5 : 1
    for (let ra = 0; ra < 360; ra += gridStep) {
      ctx.beginPath()
      for (let d = -90; d <= 90; d += 3) {
        const p = project(ra, d, W, H)
        d === -90 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
    }
    for (let dec = -80; dec <= 80; dec += gridStep) {
      ctx.beginPath()
      for (let ra = 0; ra <= 360; ra += 4) {
        const p = project(ra, dec, W, H)
        ra === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)
      }
      ctx.stroke()
    }

    // Draw stars
    starsRef.current.forEach(star => {
      ctx.globalAlpha = star.alpha

      // Glow halo for brighter stars
      if (star.r > 1.2) {
        const glow = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, star.r * 4.5)
        glow.addColorStop(0,   star.color + 'cc')
        glow.addColorStop(0.3, star.color + '44')
        glow.addColorStop(1,   star.color + '00')
        ctx.fillStyle = glow
        ctx.beginPath()
        ctx.arc(star.x, star.y, star.r * 4.5, 0, Math.PI * 2)
        ctx.fill()
      }

      // Star core
      ctx.fillStyle = star.color
      ctx.beginPath()
      ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2)
      ctx.fill()

      // Diffraction spike for brightest
      if (star.r > 2.2) {
        ctx.strokeStyle = star.color + '55'
        ctx.lineWidth = 0.5
        const sp = star.r * 3.5
        ctx.beginPath()
        ctx.moveTo(star.x - sp, star.y); ctx.lineTo(star.x + sp, star.y)
        ctx.moveTo(star.x, star.y - sp); ctx.lineTo(star.x, star.y + sp)
        ctx.stroke()
      }
    })
    ctx.globalAlpha = 1

    // Popular target markers (if in view)
    ctx.font = `${Math.max(10, 11)}px Inter, -apple-system, sans-serif`
    POPULAR_TARGETS.forEach(t => {
      const p = project(t.ra, t.dec, W, H)
      if (p.x < -30 || p.x > W + 30 || p.y < -30 || p.y > H + 30) return
      // Diamond marker
      ctx.strokeStyle = 'rgba(255,196,64,0.85)'
      ctx.fillStyle   = 'rgba(255,196,64,0.2)'
      ctx.lineWidth   = 1
      ctx.beginPath()
      ctx.moveTo(p.x, p.y - 6)
      ctx.lineTo(p.x + 5, p.y)
      ctx.lineTo(p.x, p.y + 6)
      ctx.lineTo(p.x - 5, p.y)
      ctx.closePath()
      ctx.fill(); ctx.stroke()
      // Label
      ctx.fillStyle = 'rgba(255,210,80,0.9)'
      ctx.fillText(t.name, p.x + 8, p.y + 4)
    })

    // Selected region
    const sel = project(centerRA, centerDec, W, H)
    const rPx = (radiusDeg / fovDeg) * Math.min(W, H)

    // Area fill
    ctx.fillStyle = 'rgba(26,115,232,0.07)'
    ctx.beginPath(); ctx.arc(sel.x, sel.y, rPx, 0, Math.PI * 2); ctx.fill()

    // Ring
    ctx.strokeStyle = 'rgba(26,115,232,0.9)'
    ctx.lineWidth   = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath(); ctx.arc(sel.x, sel.y, rPx, 0, Math.PI * 2); ctx.stroke()
    ctx.setLineDash([])

    // Centre crosshair
    ctx.strokeStyle = 'rgba(26,115,232,0.75)'
    ctx.lineWidth   = 1
    const cs = 9
    ctx.beginPath()
    ctx.moveTo(sel.x - cs, sel.y); ctx.lineTo(sel.x - 3, sel.y)
    ctx.moveTo(sel.x + 3,  sel.y); ctx.lineTo(sel.x + cs, sel.y)
    ctx.moveTo(sel.x, sel.y - cs); ctx.lineTo(sel.x, sel.y - 3)
    ctx.moveTo(sel.x, sel.y + 3);  ctx.lineTo(sel.x, sel.y + cs)
    ctx.stroke()

    // Radius label
    ctx.fillStyle = 'rgba(26,115,232,0.9)'
    ctx.font      = '10px Inter, sans-serif'
    ctx.fillText(`r = ${radiusDeg}°`, sel.x + rPx + 4, sel.y - 3)
  }, [centerRA, centerDec, fovDeg, radiusDeg, project])

  // ── sizing ────────────────────────────────────────────────────────────────

  const resize = useCallback(() => {
    const canvas = canvasRef.current
    const div    = containerRef.current
    if (!canvas || !div) return
    const dpr = window.devicePixelRatio || 1
    const W   = div.clientWidth
    const H   = div.clientHeight
    if (W === 0 || H === 0) return
    canvas.width  = W * dpr
    canvas.height = H * dpr
    canvas.style.width  = `${W}px`
    canvas.style.height = `${H}px`
    const ctx = canvas.getContext('2d')
    if (ctx) ctx.scale(dpr, dpr)
    sizeRef.current = { w: W, h: H }
    starsRef.current = genStars(centerRA, centerDec, W, H)
    draw()
  }, [centerRA, centerDec, draw])

  // Initial sizing
  useEffect(() => { resize() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Redraw when state changes
  useEffect(() => {
    const { w: W, h: H } = sizeRef.current
    if (W > 0 && H > 0) {
      starsRef.current = genStars(centerRA, centerDec, W, H)
      draw()
    }
  }, [centerRA, centerDec, fovDeg, draw])

  // ResizeObserver for layout changes
  useEffect(() => {
    const div = containerRef.current
    if (!div) return
    const obs = new ResizeObserver(resize)
    obs.observe(div)
    return () => obs.disconnect()
  }, [resize])

  // ── interactions ──────────────────────────────────────────────────────────

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const { ra, dec } = unproject(e.clientX - rect.left, e.clientY - rect.top, rect.width, rect.height)
    const raR = +ra.toFixed(4), decR = +dec.toFixed(4)
    setCenterRA(raR); setCenterDec(decR)
    setManualRA(String(raR)); setManualDec(String(decR))
    onSelect?.(raR, decR)
  }, [unproject, onSelect])

  // Wheel zoom — must use native listener with passive:false to call preventDefault
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const handler = (e: WheelEvent) => {
      e.preventDefault()
      setFovDeg(f => Math.max(0.5, Math.min(90, f * (e.deltaY > 0 ? 1.15 : 0.87))))
    }
    canvas.addEventListener('wheel', handler, { passive: false })
    return () => canvas.removeEventListener('wheel', handler)
  }, [])

  function handleManualGo() {
    const ra = parseFloat(manualRA), dec = parseFloat(manualDec)
    if (isNaN(ra) || isNaN(dec) || ra < 0 || ra >= 360 || dec < -90 || dec > 90) return
    setCenterRA(ra); setCenterDec(dec); onSelect?.(ra, dec)
  }

  function goTo(t: typeof POPULAR_TARGETS[number]) {
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
          <span className="text-xs text-[#9ca3af] font-mono">
            {centerRA.toFixed(3)}° / {centerDec >= 0 ? '+' : ''}{centerDec.toFixed(3)}°
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={() => setFovDeg(f => Math.max(0.5, f * 0.6))}
            title="Zoom in"
            className="w-7 h-7 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={() => setFovDeg(f => Math.min(90, f / 0.6))}
            title="Zoom out"
            className="w-7 h-7 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
            <ZoomOut className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative bg-[#060914] cursor-crosshair" style={{ height }}>
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />
        <div className="absolute top-2 right-2 text-[10px] font-mono text-white/30 select-none pointer-events-none">
          {fovDeg.toFixed(1)}° FoV · click to select · scroll to zoom
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-[#e5e7eb] space-y-3">
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">RA (deg)</label>
            <input className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualRA} onChange={e => setManualRA(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()} placeholder="0–360" />
          </div>
          <div className="flex-1">
            <label className="text-xs text-[#5f6368] mb-1 block">Dec (deg)</label>
            <input className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualDec} onChange={e => setManualDec(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()} placeholder="-90 to +90" />
          </div>
          <button onClick={handleManualGo} className="btn btn-primary text-sm px-4 py-2">Go</button>
        </div>

        <button onClick={() => onSelect?.(centerRA, centerDec)}
          className="w-full flex items-center justify-center gap-2 btn btn-outline text-sm py-2">
          <MapPin className="w-4 h-4" />
          Use this region — RA {centerRA.toFixed(2)}°, Dec {centerDec.toFixed(2)}°
        </button>

        <div>
          <p className="text-xs text-[#5f6368] mb-2">Popular targets:</p>
          <div className="flex flex-wrap gap-2">
            {POPULAR_TARGETS.map(t => (
              <button key={t.name} onClick={() => goTo(t)} title={t.description}
                className="text-xs px-2.5 py-1 rounded-full bg-[#f1f3f4] text-[#202124] hover:bg-[#e8eaed] transition-colors">
                {t.name}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
