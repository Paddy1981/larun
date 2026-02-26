'use client'

/**
 * LightCurveViewer — SVG-based interactive light curve plotter
 * Renders flux vs time with hover crosshair, model result overlays, and fold view.
 */

import { useState, useRef, useCallback } from 'react'
import { ZoomIn, ZoomOut, RefreshCw, GitBranch } from 'lucide-react'

interface LightCurveViewerProps {
  times: number[]
  flux: number[]
  fluxErr?: number[]
  period?: number          // days — enables phase-fold toggle
  title?: string
  highlightTransits?: { t0: number; duration: number }[]
  height?: number
  width?: number
}

const PADDING = { top: 20, right: 20, bottom: 45, left: 55 }

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v))
}

export function LightCurveViewer({
  times,
  flux,
  fluxErr,
  period,
  title,
  highlightTransits,
  height = 300,
  width: propWidth,
}: LightCurveViewerProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const [folded, setFolded] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [hoverIdx, setHoverIdx] = useState<number | null>(null)
  const [svgWidth, setSvgWidth] = useState(propWidth ?? 600)

  // Observe container resize
  const containerRef = useCallback((el: HTMLDivElement | null) => {
    if (!el) return
    const ro = new ResizeObserver(entries => {
      const w = entries[0].contentRect.width
      if (w > 0) setSvgWidth(w)
    })
    ro.observe(el)
  }, [])

  if (!times || times.length === 0) {
    return (
      <div
        className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm flex items-center justify-center"
        style={{ height }}
      >
        <span className="text-sm text-[#5f6368]">No light curve data</span>
      </div>
    )
  }

  // Compute x/y values
  const xVals = folded && period
    ? times.map(t => ((t % period) + period) % period)
    : times

  const plotW = svgWidth - PADDING.left - PADDING.right
  const plotH = height - PADDING.top - PADDING.bottom

  // Visible x range after zoom
  const xMin = Math.min(...xVals)
  const xMax = Math.max(...xVals)
  const xRange = (xMax - xMin) / zoom
  const xCenter = (xMin + xMax) / 2
  const xLo = xCenter - xRange / 2
  const xHi = xCenter + xRange / 2

  const visible = xVals.map((x, i) => x >= xLo && x <= xHi ? i : -1).filter(i => i >= 0)
  const visFlux = visible.map(i => flux[i])
  const yLo = Math.min(...visFlux) - 0.001
  const yHi = Math.max(...visFlux) + 0.001
  const yRange = yHi - yLo

  function toSvgX(x: number) {
    return PADDING.left + ((x - xLo) / (xHi - xLo)) * plotW
  }
  function toSvgY(y: number) {
    return PADDING.top + (1 - (y - yLo) / yRange) * plotH
  }

  // Build polyline points
  const sorted = [...visible].sort((a, b) => xVals[a] - xVals[b])
  const polyline = sorted
    .map(i => `${toSvgX(xVals[i]).toFixed(1)},${toSvgY(flux[i]).toFixed(1)}`)
    .join(' ')

  // Error bars (sparse: every 5th point to keep performance)
  const errorBars = fluxErr
    ? sorted.filter((_, idx) => idx % 5 === 0).map(i => {
        const cx = toSvgX(xVals[i])
        const cy = toSvgY(flux[i])
        const halfErr = (fluxErr[i] / yRange) * plotH
        return { i, cx, cy, halfErr }
      })
    : []

  // Transit highlights
  const transitRects = (highlightTransits ?? []).map(({ t0, duration }) => {
    const x1 = toSvgX(t0 - duration / 2)
    const x2 = toSvgX(t0 + duration / 2)
    return { x: x1, width: Math.max(2, x2 - x1) }
  })

  // Hover crosshair
  function handleMouseMove(e: React.MouseEvent<SVGSVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect()
    if (!rect) return
    const mouseX = e.clientX - rect.left - PADDING.left
    const frac = clamp(mouseX / plotW, 0, 1)
    const targetX = xLo + frac * (xHi - xLo)
    let closest = sorted[0]
    let bestDist = Infinity
    for (const i of sorted) {
      const d = Math.abs(xVals[i] - targetX)
      if (d < bestDist) { bestDist = d; closest = i }
    }
    setHoverIdx(closest)
  }

  // Axis ticks
  const xTicks = 5
  const yTicks = 4

  return (
    <div className="bg-white rounded-2xl border border-[#e5e7eb] shadow-sm overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#e5e7eb] flex items-center justify-between">
        <span className="text-sm font-medium text-[#202124]">
          {title ?? 'Light Curve'}{folded && period ? ` — Folded (P = ${period.toFixed(3)}d)` : ''}
        </span>
        <div className="flex items-center gap-2">
          {period && (
            <button
              onClick={() => setFolded(f => !f)}
              title="Phase fold"
              className={`p-1.5 rounded-lg transition-colors ${folded ? 'bg-[#e8f0fe] text-[#1a73e8]' : 'hover:bg-[#f1f3f4] text-[#5f6368]'}`}
            >
              <GitBranch className="w-4 h-4" />
            </button>
          )}
          <button onClick={() => setZoom(z => Math.min(z * 1.5, 8))} title="Zoom in"
            className="p-1.5 rounded-lg hover:bg-[#f1f3f4] text-[#5f6368] transition-colors">
            <ZoomIn className="w-4 h-4" />
          </button>
          <button onClick={() => setZoom(z => Math.max(z / 1.5, 1))} title="Zoom out"
            className="p-1.5 rounded-lg hover:bg-[#f1f3f4] text-[#5f6368] transition-colors">
            <ZoomOut className="w-4 h-4" />
          </button>
          <button onClick={() => setZoom(1)} title="Reset zoom"
            className="p-1.5 rounded-lg hover:bg-[#f1f3f4] text-[#5f6368] transition-colors">
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* SVG plot */}
      <div ref={containerRef} style={{ width: '100%' }}>
        <svg
          ref={svgRef}
          width={svgWidth}
          height={height}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverIdx(null)}
          style={{ display: 'block' }}
        >
          {/* Plot background */}
          <rect
            x={PADDING.left} y={PADDING.top}
            width={plotW} height={plotH}
            fill="#fafafa" stroke="#e5e7eb" strokeWidth={1}
          />

          {/* Transit highlights */}
          {transitRects.map((r, idx) => (
            <rect
              key={idx}
              x={Math.max(PADDING.left, r.x)}
              y={PADDING.top}
              width={Math.min(r.width, plotW)}
              height={plotH}
              fill="rgba(26,115,232,0.1)"
              stroke="rgba(26,115,232,0.3)"
              strokeWidth={1}
            />
          ))}

          {/* X gridlines */}
          {Array.from({ length: xTicks }, (_, i) => {
            const xVal = xLo + ((i + 1) / (xTicks + 1)) * (xHi - xLo)
            const sx = toSvgX(xVal)
            return (
              <line key={i} x1={sx} y1={PADDING.top} x2={sx} y2={PADDING.top + plotH}
                stroke="#e5e7eb" strokeWidth={1} strokeDasharray="3,3" />
            )
          })}

          {/* Y gridlines */}
          {Array.from({ length: yTicks }, (_, i) => {
            const yVal = yLo + ((i + 1) / (yTicks + 1)) * yRange
            const sy = toSvgY(yVal)
            return (
              <line key={i} x1={PADDING.left} y1={sy} x2={PADDING.left + plotW} y2={sy}
                stroke="#e5e7eb" strokeWidth={1} strokeDasharray="3,3" />
            )
          })}

          {/* Error bars */}
          {errorBars.map(({ i, cx, cy, halfErr }) => (
            <line key={i} x1={cx} y1={cy - halfErr} x2={cx} y2={cy + halfErr}
              stroke="#9ca3af" strokeWidth={1} />
          ))}

          {/* Light curve polyline */}
          <polyline
            points={polyline}
            fill="none"
            stroke="#1a73e8"
            strokeWidth={1.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* Data points (sparse) */}
          {sorted.filter((_, idx) => idx % 3 === 0).map(i => (
            <circle
              key={i}
              cx={toSvgX(xVals[i])}
              cy={toSvgY(flux[i])}
              r={hoverIdx === i ? 4 : 1.5}
              fill={hoverIdx === i ? '#202124' : '#1a73e8'}
            />
          ))}

          {/* Hover crosshair */}
          {hoverIdx !== null && (() => {
            const cx = toSvgX(xVals[hoverIdx])
            const cy = toSvgY(flux[hoverIdx])
            return (
              <>
                <line x1={cx} y1={PADDING.top} x2={cx} y2={PADDING.top + plotH}
                  stroke="#202124" strokeWidth={1} strokeDasharray="4,4" opacity={0.5} />
                <line x1={PADDING.left} y1={cy} x2={PADDING.left + plotW} y2={cy}
                  stroke="#202124" strokeWidth={1} strokeDasharray="4,4" opacity={0.5} />
                <rect
                  x={cx + 6} y={cy - 22}
                  width={130} height={40}
                  rx={4} fill="white"
                  stroke="#e5e7eb" strokeWidth={1}
                />
                <text x={cx + 10} y={cy - 7} fontSize={10} fill="#202124">
                  t = {xVals[hoverIdx].toFixed(3)}{folded ? 'd (phase)' : ' d'}
                </text>
                <text x={cx + 10} y={cy + 8} fontSize={10} fill="#1a73e8">
                  flux = {flux[hoverIdx].toFixed(5)}
                </text>
              </>
            )
          })()}

          {/* X-axis ticks + labels */}
          {Array.from({ length: xTicks + 2 }, (_, i) => {
            const xVal = xLo + (i / (xTicks + 1)) * (xHi - xLo)
            const sx = toSvgX(xVal)
            return (
              <g key={i}>
                <line x1={sx} y1={PADDING.top + plotH} x2={sx} y2={PADDING.top + plotH + 5} stroke="#9ca3af" />
                <text x={sx} y={PADDING.top + plotH + 16} fontSize={9} textAnchor="middle" fill="#5f6368">
                  {xVal.toFixed(1)}
                </text>
              </g>
            )
          })}

          {/* Y-axis ticks + labels */}
          {Array.from({ length: yTicks + 2 }, (_, i) => {
            const yVal = yLo + (i / (yTicks + 1)) * yRange
            const sy = toSvgY(yVal)
            return (
              <g key={i}>
                <line x1={PADDING.left - 5} y1={sy} x2={PADDING.left} y2={sy} stroke="#9ca3af" />
                <text x={PADDING.left - 8} y={sy + 3} fontSize={9} textAnchor="end" fill="#5f6368">
                  {yVal.toFixed(4)}
                </text>
              </g>
            )
          })}

          {/* Axis labels */}
          <text
            x={PADDING.left + plotW / 2}
            y={height - 5}
            fontSize={11} textAnchor="middle" fill="#5f6368"
          >
            {folded && period ? 'Phase (days)' : 'Time (days)'}
          </text>
          <text
            transform={`translate(12,${PADDING.top + plotH / 2}) rotate(-90)`}
            fontSize={11} textAnchor="middle" fill="#5f6368"
          >
            Normalized Flux
          </text>
        </svg>
      </div>

      {/* Stats bar */}
      <div className="px-4 py-2 border-t border-[#e5e7eb] flex gap-6 text-xs text-[#5f6368]">
        <span>{times.length} points</span>
        {period && <span>P = {period.toFixed(4)} d</span>}
        <span>Δf = {(yHi - yLo).toExponential(2)}</span>
        {zoom > 1 && <span>Zoom {zoom.toFixed(1)}×</span>}
      </div>
    </div>
  )
}
