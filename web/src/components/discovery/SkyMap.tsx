'use client'

/**
 * SkyMap — Canvas-based interactive sky chart
 * Screen-space star generation: always dense, loads instantly, no CDN.
 * Includes: N/E arrows, scale bar, constellation hint, HMS/DMS display,
 * radius in arcminutes, improved coordinate labels.
 */

import { useEffect, useRef, useState, useCallback } from 'react'
import { MapPin, Target, ZoomIn, ZoomOut, Info } from 'lucide-react'

interface SkyMapProps {
  onSelect?: (ra: number, dec: number) => void
  initialRA?: number
  initialDec?: number
  radiusDeg?: number
  height?: number
}

const POPULAR_TARGETS = [
  { name: 'Pleiades',       ra: 56.75,  dec: 24.12,  description: 'Open cluster, M45 in Taurus' },
  { name: 'Galactic Bulge', ra: 266.42, dec: -29.01, description: 'OGLE microlensing field, Sagittarius' },
  { name: 'Kepler Field',   ra: 291.0,  dec: 44.5,   description: 'Original Kepler field of view, Cygnus' },
  { name: 'TESS CVZ South', ra: 90.0,   dec: -66.0,  description: 'TESS continuous viewing zone, Pictor' },
  { name: 'LMC',            ra: 80.89,  dec: -69.76, description: 'Large Magellanic Cloud, Dorado' },
  { name: 'Orion OB1',      ra: 83.82,  dec: -5.39,  description: 'Active star-forming region, Orion' },
]

// Rough constellation centers — used to show "you are in Taurus" style hint
const CONSTELLATION_CENTERS = [
  { name: 'Orion',       ra: 83,  dec: 5   },
  { name: 'Taurus',      ra: 68,  dec: 20  },
  { name: 'Gemini',      ra: 113, dec: 25  },
  { name: 'Leo',         ra: 165, dec: 15  },
  { name: 'Virgo',       ra: 196, dec: -4  },
  { name: 'Scorpius',    ra: 253, dec: -26 },
  { name: 'Sagittarius', ra: 285, dec: -25 },
  { name: 'Cygnus',      ra: 309, dec: 42  },
  { name: 'Aquila',      ra: 297, dec: 5   },
  { name: 'Andromeda',   ra: 11,  dec: 41  },
  { name: 'Perseus',     ra: 50,  dec: 45  },
  { name: 'Auriga',      ra: 90,  dec: 42  },
  { name: 'Ursa Major',  ra: 165, dec: 56  },
  { name: 'Bootes',      ra: 220, dec: 30  },
  { name: 'Hercules',    ra: 255, dec: 30  },
  { name: 'Lyra',        ra: 284, dec: 36  },
  { name: 'Aquarius',    ra: 336, dec: -10 },
  { name: 'Pisces',      ra: 16,  dec: 13  },
  { name: 'Aries',       ra: 35,  dec: 20  },
  { name: 'Cetus',       ra: 25,  dec: -10 },
  { name: 'Eridanus',    ra: 50,  dec: -30 },
  { name: 'Canis Major', ra: 104, dec: -22 },
  { name: 'Hydra',       ra: 155, dec: -17 },
  { name: 'Centaurus',   ra: 205, dec: -47 },
  { name: 'Ophiuchus',   ra: 257, dec: -5  },
  { name: 'Capricornus', ra: 321, dec: -18 },
  { name: 'Pegasus',     ra: 340, dec: 20  },
  { name: 'Draco',       ra: 260, dec: 65  },
  { name: 'Lepus',       ra: 81,  dec: -19 },
]

function angSep(ra1: number, dec1: number, ra2: number, dec2: number): number {
  const toRad = Math.PI / 180
  const d1 = dec1 * toRad, d2 = dec2 * toRad
  const dra = (ra2 - ra1) * toRad
  return Math.acos(
    Math.min(1, Math.sin(d1) * Math.sin(d2) + Math.cos(d1) * Math.cos(d2) * Math.cos(dra))
  ) / toRad
}

function nearestConstellation(ra: number, dec: number): string {
  let best = CONSTELLATION_CENTERS[0], bestDist = Infinity
  for (const c of CONSTELLATION_CENTERS) {
    const d = angSep(ra, dec, c.ra, c.dec)
    if (d < bestDist) { bestDist = d; best = c }
  }
  return best.name
}

/** Decimal RA degrees → h m s string */
function raToHMS(ra: number): string {
  const total = ((ra % 360) + 360) % 360
  const h = Math.floor(total / 15)
  const mFrac = (total / 15 - h) * 60
  const m = Math.floor(mFrac)
  const s = Math.round((mFrac - m) * 60)
  return `${h}h ${m.toString().padStart(2, '0')}m ${s.toString().padStart(2, '0')}s`
}

/** Decimal Dec degrees → ° ′ ″ string */
function decToDMS(dec: number): string {
  const sign = dec < 0 ? '-' : '+'
  const abs = Math.abs(dec)
  const d = Math.floor(abs)
  const mFrac = (abs - d) * 60
  const m = Math.floor(mFrac)
  const s = Math.round((mFrac - m) * 60)
  return `${sign}${d}d ${m.toString().padStart(2, '0')}m ${s.toString().padStart(2, '0')}s`
}

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

function genStars(seedRA: number, seedDec: number, W: number, H: number): Star[] {
  const qRA  = Math.round(seedRA  / 8) * 8
  const qDec = Math.round(seedDec / 8) * 8
  const seed = (qRA * 1301 + qDec * 4507 + 0xdeadbeef) | 0
  const rand = rng32(seed)
  const stars: Star[] = []
  for (let i = 0; i < 520; i++) {
    const mag  = rand() * rand()
    const r    = 0.35 + (1 - mag) * 0.55
    const a    = 0.15 + mag * 0.5
    const type = rand()
    const color = type < 0.06 ? '#ffa060' : type < 0.12 ? '#a0c0ff' : '#d8e8ff'
    stars.push({ x: rand() * W, y: rand() * H, r, color, alpha: a })
  }
  for (let i = 0; i < 160; i++) {
    const mag  = rand() * rand() * rand()
    const r    = 0.7 + (1 - mag) * 1.1
    const a    = 0.45 + mag * 0.45
    const type = rand()
    const color = type < 0.05 ? '#ff8844' : type < 0.10 ? '#88aaff' : '#e8f0ff'
    stars.push({ x: rand() * W, y: rand() * H, r, color, alpha: a })
  }
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
  const [showHelp,  setShowHelp]  = useState(false)

  const constellation = nearestConstellation(centerRA, centerDec)

  // ── coordinate projection ─────────────────────────────────────────────────

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

    // Background
    const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, Math.max(W, H) * 0.75)
    bg.addColorStop(0,   '#0d1526')
    bg.addColorStop(0.6, '#080e1c')
    bg.addColorStop(1,   '#04090f')
    ctx.fillStyle = bg
    ctx.fillRect(0, 0, W, H)

    // Milky Way glow
    const galIntensity = Math.max(0, 1 - Math.abs(centerDec + 5 * Math.sin(centerRA * 0.0174)) / 35)
    if (galIntensity > 0.05) {
      const mw = ctx.createLinearGradient(0, H * 0.1, W, H * 0.9)
      mw.addColorStop(0,    'rgba(60,80,140,0)')
      mw.addColorStop(0.35, `rgba(60,80,140,${galIntensity * 0.07})`)
      mw.addColorStop(0.5,  `rgba(80,100,160,${galIntensity * 0.12})`)
      mw.addColorStop(0.65, `rgba(60,80,140,${galIntensity * 0.07})`)
      mw.addColorStop(1,    'rgba(60,80,140,0)')
      ctx.fillStyle = mw
      ctx.fillRect(0, 0, W, H)
    }

    // RA/Dec grid
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

    // Grid labels
    ctx.fillStyle = 'rgba(255,255,255,0.20)'
    ctx.font = '9px Inter, sans-serif'
    if (fovDeg <= 40) {
      // Dec labels on left edge
      for (let dec = -80; dec <= 80; dec += gridStep) {
        const p = project(centerRA, dec, W, H)
        if (p.y > 12 && p.y < H - 10) {
          ctx.fillText(`${dec > 0 ? '+' : ''}${dec}°`, 4, p.y + 3)
        }
      }
      // RA hour labels along top
      for (let ra = 0; ra < 360; ra += gridStep) {
        const p = project(ra, centerDec, W, H)
        if (p.x > 20 && p.x < W - 20) {
          ctx.fillText(`${Math.floor(ra / 15)}h`, p.x - 5, 12)
        }
      }
    }

    // Stars
    starsRef.current.forEach(star => {
      ctx.globalAlpha = star.alpha
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
      ctx.fillStyle = star.color
      ctx.beginPath()
      ctx.arc(star.x, star.y, star.r, 0, Math.PI * 2)
      ctx.fill()
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

    // Popular target markers
    ctx.font = '11px Inter, sans-serif'
    POPULAR_TARGETS.forEach(t => {
      const p = project(t.ra, t.dec, W, H)
      if (p.x < -30 || p.x > W + 30 || p.y < -30 || p.y > H + 30) return
      ctx.strokeStyle = 'rgba(255,196,64,0.85)'
      ctx.fillStyle   = 'rgba(255,196,64,0.2)'
      ctx.lineWidth   = 1
      ctx.beginPath()
      ctx.moveTo(p.x, p.y - 6); ctx.lineTo(p.x + 5, p.y)
      ctx.lineTo(p.x, p.y + 6); ctx.lineTo(p.x - 5, p.y)
      ctx.closePath(); ctx.fill(); ctx.stroke()
      ctx.fillStyle = 'rgba(255,210,80,0.9)'
      ctx.fillText(t.name, p.x + 8, p.y + 4)
    })

    // Selected region circle
    const sel = project(centerRA, centerDec, W, H)
    const rPx = (radiusDeg / fovDeg) * Math.min(W, H)

    ctx.fillStyle = 'rgba(26,115,232,0.07)'
    ctx.beginPath(); ctx.arc(sel.x, sel.y, rPx, 0, Math.PI * 2); ctx.fill()

    ctx.strokeStyle = 'rgba(26,115,232,0.9)'
    ctx.lineWidth   = 1.5
    ctx.setLineDash([5, 4])
    ctx.beginPath(); ctx.arc(sel.x, sel.y, rPx, 0, Math.PI * 2); ctx.stroke()
    ctx.setLineDash([])

    // Crosshair at centre
    ctx.strokeStyle = 'rgba(26,115,232,0.75)'
    ctx.lineWidth   = 1
    const cs = 9
    ctx.beginPath()
    ctx.moveTo(sel.x - cs, sel.y); ctx.lineTo(sel.x - 3, sel.y)
    ctx.moveTo(sel.x + 3,  sel.y); ctx.lineTo(sel.x + cs, sel.y)
    ctx.moveTo(sel.x, sel.y - cs); ctx.lineTo(sel.x, sel.y - 3)
    ctx.moveTo(sel.x, sel.y + 3);  ctx.lineTo(sel.x, sel.y + cs)
    ctx.stroke()

    // Radius label with arcminutes
    const arcmin = Math.round(radiusDeg * 60)
    const rLabel = arcmin >= 60
      ? `r = ${radiusDeg}° (${(arcmin / 60).toFixed(1)}h region)`
      : `r = ${radiusDeg}° (${arcmin}' = ${(arcmin / 30).toFixed(1)}x Moon)`
    ctx.fillStyle = 'rgba(100,160,255,0.9)'
    ctx.font      = '10px Inter, sans-serif'
    ctx.fillText(rLabel, sel.x + rPx + 5, sel.y - 4)

    // ── N / E compass (bottom-right) ──────────────────────────────────────
    // Astronomy: N = up (Dec+), E = LEFT (RA increases east, but maps left)
    const ax = W - 38, ay = H - 36, al = 18
    ctx.lineWidth = 1.5
    // N arrow
    ctx.strokeStyle = 'rgba(255,255,255,0.50)'
    ctx.fillStyle   = 'rgba(255,255,255,0.50)'
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax, ay - al); ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(ax, ay - al - 3); ctx.lineTo(ax - 3, ay - al + 4); ctx.lineTo(ax + 3, ay - al + 4)
    ctx.closePath(); ctx.fill()
    ctx.font = '9px Inter, sans-serif'
    ctx.fillText('N', ax - 3, ay - al - 6)
    // E arrow (left)
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax - al, ay); ctx.stroke()
    ctx.beginPath()
    ctx.moveTo(ax - al - 3, ay); ctx.lineTo(ax - al + 4, ay - 3); ctx.lineTo(ax - al + 4, ay + 3)
    ctx.closePath(); ctx.fill()
    ctx.fillText('E', ax - al - 13, ay + 3)

    // ── Scale bar (bottom-left) ───────────────────────────────────────────
    const niceScales = [0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30]
    const degPerPx   = fovDeg / Math.min(W, H)
    const targetDeg  = W * 0.18 * degPerPx
    const scaleDeg   = niceScales.reduce((prev, cur) =>
      Math.abs(cur - targetDeg) < Math.abs(prev - targetDeg) ? cur : prev
    )
    const scalePx  = scaleDeg / degPerPx
    const sbX = 16, sbY = H - 12
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'
    ctx.lineWidth   = 1.5
    ctx.beginPath()
    ctx.moveTo(sbX, sbY);        ctx.lineTo(sbX + scalePx, sbY)
    ctx.moveTo(sbX, sbY - 3);   ctx.lineTo(sbX, sbY + 3)
    ctx.moveTo(sbX + scalePx, sbY - 3); ctx.lineTo(sbX + scalePx, sbY + 3)
    ctx.stroke()
    ctx.fillStyle = 'rgba(255,255,255,0.55)'
    ctx.font      = '9px Inter, sans-serif'
    const scaleLabel = scaleDeg < 1
      ? `${Math.round(scaleDeg * 60)}'`
      : `${scaleDeg}\u00b0`
    ctx.fillText(scaleLabel, sbX + scalePx / 2 - 6, sbY - 5)

    // Constellation watermark (faint, near compass)
    ctx.fillStyle = 'rgba(255,255,255,0.18)'
    ctx.font      = '10px Inter, sans-serif'
    ctx.fillText(constellation, W - 38 - 48, H - 58)

  }, [centerRA, centerDec, fovDeg, radiusDeg, project, constellation])

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

  useEffect(() => { resize() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const { w: W, h: H } = sizeRef.current
    if (W > 0 && H > 0) {
      starsRef.current = genStars(centerRA, centerDec, W, H)
      draw()
    }
  }, [centerRA, centerDec, fovDeg, draw])

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

      {/* Header bar */}
      <div className="px-4 py-3 border-b border-[#e5e7eb]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 min-w-0">
            <Target className="w-4 h-4 text-[#1a73e8] shrink-0" />
            <span className="font-medium text-[#202124] text-sm">Sky Region</span>
            <span className="hidden sm:inline text-xs text-[#9ca3af] font-mono truncate">
              {raToHMS(centerRA)} / {decToDMS(centerDec)}
            </span>
            <span className="text-[10px] text-[#bdbdbd] italic truncate">· {constellation}</span>
          </div>
          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={() => setShowHelp(v => !v)}
              title="What are RA and Dec?"
              className={`w-7 h-7 rounded flex items-center justify-center transition-colors ${
                showHelp ? 'bg-[#e8f0fe] text-[#1a73e8]' : 'text-[#9ca3af] hover:bg-[#f1f3f4]'
              }`}
            >
              <Info className="w-3.5 h-3.5" />
            </button>
            <button onClick={() => setFovDeg(f => Math.max(0.5, f * 0.6))} title="Zoom in"
              className="w-7 h-7 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
              <ZoomIn className="w-4 h-4" />
            </button>
            <button onClick={() => setFovDeg(f => Math.min(90, f / 0.6))} title="Zoom out"
              className="w-7 h-7 rounded flex items-center justify-center text-[#5f6368] hover:bg-[#f1f3f4] transition-colors">
              <ZoomOut className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Coordinate explainer */}
        {showHelp && (
          <div className="mt-2 bg-[#f8f9fa] rounded-xl p-3 text-xs text-[#5f6368] space-y-1.5 border border-[#e5e7eb]">
            <p>
              <strong className="text-[#202124]">Right Ascension (RA)</strong> — the sky&apos;s east-west coordinate,
              like longitude. Runs 0° to 360° (or 0h to 24h). On a sky map, east is to the left.
            </p>
            <p>
              <strong className="text-[#202124]">Declination (Dec)</strong> — the sky&apos;s north-south coordinate,
              like latitude. 0° = celestial equator · +90° = north pole · −90° = south pole.
            </p>
            <p>
              <strong className="text-[#202124]">Search radius</strong> — how wide a circle to scan.
              0.5° = 30 arcminutes ≈ one full Moon diameter. The blue dashed circle on the map shows your selected area.
            </p>
            <p className="text-[#9ca3af]">
              Tip: click any point on the map, or type coordinates and press Go, or choose a quick target below.
            </p>
          </div>
        )}
      </div>

      {/* Canvas */}
      <div ref={containerRef} className="relative bg-[#060914] cursor-crosshair" style={{ height }}>
        <canvas
          ref={canvasRef}
          onClick={handleClick}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />
        {/* FoV badge */}
        <div className="absolute top-2 right-2 bg-black/40 rounded px-2 py-0.5 text-[10px] font-mono text-white/40 select-none pointer-events-none">
          {fovDeg.toFixed(1)}° FoV · click · scroll to zoom
        </div>
        {/* HMS/DMS centre overlay */}
        <div className="absolute bottom-3 left-1/2 -translate-x-1/2 bg-black/50 rounded-lg px-3 py-1 text-[10px] font-mono text-white/60 select-none pointer-events-none whitespace-nowrap">
          {raToHMS(centerRA)} / {decToDMS(centerDec)}
        </div>
      </div>

      {/* Controls */}
      <div className="p-4 border-t border-[#e5e7eb] space-y-3">

        {/* RA / Dec inputs */}
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <label className="text-xs font-medium text-[#202124] block mb-0.5">
              RA — Right Ascension
            </label>
            <p className="text-[10px] text-[#9ca3af] mb-1.5">
              0° to 360° (sky longitude)
              {' · '}
              <span className="font-mono text-[#5f6368]">
                {raToHMS(isNaN(parseFloat(manualRA)) ? centerRA : parseFloat(manualRA))}
              </span>
            </p>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualRA}
              onChange={e => setManualRA(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()}
              placeholder="0 – 360  e.g. 83.82"
            />
          </div>
          <div className="flex-1">
            <label className="text-xs font-medium text-[#202124] block mb-0.5">
              Dec — Declination
            </label>
            <p className="text-[10px] text-[#9ca3af] mb-1.5">
              -90° to +90° (sky latitude)
              {' · '}
              <span className="font-mono text-[#5f6368]">
                {decToDMS(isNaN(parseFloat(manualDec)) ? centerDec : parseFloat(manualDec))}
              </span>
            </p>
            <input
              className="w-full text-sm border border-[#dadce0] rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-[#1a73e8]"
              value={manualDec}
              onChange={e => setManualDec(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleManualGo()}
              placeholder="-90 to +90  e.g. -5.39"
            />
          </div>
          <button onClick={handleManualGo} className="btn btn-primary text-sm px-4 py-2">Go</button>
        </div>

        {/* Use this region button */}
        <button
          onClick={() => onSelect?.(centerRA, centerDec)}
          className="w-full flex items-center justify-center gap-2 btn btn-outline text-sm py-2"
        >
          <MapPin className="w-4 h-4" />
          Use {constellation} region — RA {centerRA.toFixed(2)}°, Dec {centerDec >= 0 ? '+' : ''}{centerDec.toFixed(2)}°
        </button>

        {/* Quick targets */}
        <div>
          <p className="text-xs font-medium text-[#5f6368] mb-2">Quick targets:</p>
          <div className="flex flex-wrap gap-2">
            {POPULAR_TARGETS.map(t => (
              <button
                key={t.name}
                onClick={() => goTo(t)}
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
