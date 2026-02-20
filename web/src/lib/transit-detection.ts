/**
 * Transit Detection Service
 *
 * Implements the Box Least Squares (BLS) algorithm for detecting
 * periodic transit signals in light curves.
 *
 * Based on Kovács, Zucker, & Mazeh (2002) method.
 */

import { LightCurveData, TICInfo } from './mast-service';

export interface TransitParameters {
  period: number;        // Orbital period in days
  epoch: number;         // Transit epoch (BJD)
  depth: number;         // Transit depth (fractional)
  depth_ppm: number;     // Transit depth in parts per million
  duration: number;      // Transit duration in hours
  snr: number;           // Signal-to-noise ratio
  power: number;         // BLS power (detection statistic)
}

export interface BLSResult {
  bestPeriod: number;
  bestPower: number;
  periods: number[];
  powers: number[];
  transitParams: TransitParameters | null;
}

export interface VettingTest {
  flag: 'PASS' | 'FAIL' | 'WARNING';
  message: string;
  confidence: number;
}

export interface VettingResult {
  disposition: 'PLANET_CANDIDATE' | 'LIKELY_FALSE_POSITIVE' | 'INCONCLUSIVE';
  confidence: number;
  odd_even: VettingTest;
  v_shape: VettingTest;
  secondary_eclipse: VettingTest;
}

export interface DetectionResult {
  detection: boolean;
  confidence: number;
  period_days: number | null;
  depth_ppm: number | null;
  duration_hours: number | null;
  epoch_btjd: number | null;
  snr: number | null;
  vetting: VettingResult | null;
  sectors_used: number[];
  processing_time_seconds: number;
  folded_curve?: { phase: number; flux: number }[];
}

/**
 * Calculate the median of an array
 */
function median(arr: number[]): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0
    ? sorted[mid]
    : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Calculate the Median Absolute Deviation (MAD)
 */
function mad(arr: number[]): number {
  const med = median(arr);
  const deviations = arr.map((x) => Math.abs(x - med));
  return median(deviations);
}

/**
 * Calculate standard deviation
 */
function std(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const variance = arr.reduce((sum, x) => sum + (x - mean) ** 2, 0) / arr.length;
  return Math.sqrt(variance);
}

/**
 * Remove outliers using sigma clipping
 */
function sigmaClip(
  time: number[],
  flux: number[],
  sigma: number = 3
): { time: number[]; flux: number[] } {
  const med = median(flux);
  const madVal = mad(flux) * 1.4826; // Convert MAD to std

  const mask = flux.map((f) => Math.abs(f - med) < sigma * madVal);

  return {
    time: time.filter((_, i) => mask[i]),
    flux: flux.filter((_, i) => mask[i]),
  };
}

/**
 * Detrend light curve using a simple moving median
 */
function detrendLightCurve(
  time: number[],
  flux: number[],
  windowSize: number = 101
): number[] {
  const detrended: number[] = [];
  const halfWindow = Math.floor(windowSize / 2);

  for (let i = 0; i < flux.length; i++) {
    const start = Math.max(0, i - halfWindow);
    const end = Math.min(flux.length, i + halfWindow + 1);
    const windowFlux = flux.slice(start, end);
    const medianFlux = median(windowFlux);
    detrended.push(flux[i] / medianFlux);
  }

  return detrended;
}

/**
 * Bin-average downsample a time series to at most maxPoints data points.
 * Reduces BLS computation time from O(N) → O(maxPoints) without losing signal.
 */
function downsampleTimeSeries(
  time: number[],
  flux: number[],
  maxPoints = 3000
): { time: number[]; flux: number[] } {
  if (time.length <= maxPoints) return { time, flux };
  const stride = Math.ceil(time.length / maxPoints);
  const dTime: number[] = [];
  const dFlux: number[] = [];
  for (let i = 0; i < time.length; i += stride) {
    const end = Math.min(i + stride, time.length);
    let tSum = 0, fSum = 0, n = 0;
    for (let j = i; j < end; j++) { tSum += time[j]; fSum += flux[j]; n++; }
    dTime.push(tSum / n);
    dFlux.push(fSum / n);
  }
  return { time: dTime, flux: dFlux };
}

/**
 * Box Least Squares (BLS) algorithm
 *
 * Searches for periodic box-shaped dips in the light curve.
 */
export function runBLS(
  time: number[],
  flux: number[],
  options: {
    minPeriod?: number;
    maxPeriod?: number;
    periodSteps?: number;
    minTransitDuration?: number;
    maxTransitDuration?: number;
    hintPeriod?: number;   // Exact period to test with a fine-grid search
  } = {}
): BLSResult {
  const {
    minPeriod = 0.5,
    maxPeriod = 40,
    periodSteps = 500,           // 500 log-spaced periods; adequate resolution
    minTransitDuration = 0.01,   // Fraction of period (general BLS)
    maxTransitDuration = 0.15,
    hintPeriod,
  } = options;

  // Normalize flux
  const meanFlux = flux.reduce((a, b) => a + b, 0) / flux.length;
  const normFlux = flux.map((f) => f / meanFlux - 1);

  // Generate period grid (logarithmic spacing)
  const periods: number[] = [];
  const logMin = Math.log10(minPeriod);
  const logMax = Math.log10(maxPeriod);
  const logStep = (logMax - logMin) / periodSteps;

  for (let i = 0; i < periodSteps; i++) {
    periods.push(Math.pow(10, logMin + i * logStep));
  }

  const powers: number[] = [];
  let bestPower = 0;
  let bestPeriod = 0;
  let bestPhase = 0;
  let bestDuration = 0;

  // BLS search over periods
  for (const period of periods) {
    // Phase fold the data
    const phases = time.map((t) => ((t - time[0]) % period) / period);

    // Search over transit durations
    const durationSteps = 10;
    for (let d = 0; d < durationSteps; d++) {
      const transitDuration =
        minTransitDuration +
        ((maxTransitDuration - minTransitDuration) * d) / durationSteps;

      // Search over phases
      const phaseSteps = Math.floor(1 / transitDuration);
      for (let p = 0; p < phaseSteps; p++) {
        const phaseStart = p / phaseSteps;
        const phaseEnd = phaseStart + transitDuration;

        // Calculate BLS statistic
        let inTransitSum = 0;
        let inTransitCount = 0;
        let outTransitSum = 0;
        let outTransitCount = 0;

        for (let i = 0; i < phases.length; i++) {
          const phase = phases[i];
          const wrappedPhase =
            phase < phaseStart ? phase + 1 : phase;

          if (
            wrappedPhase >= phaseStart &&
            wrappedPhase < phaseStart + transitDuration
          ) {
            inTransitSum += normFlux[i];
            inTransitCount++;
          } else {
            outTransitSum += normFlux[i];
            outTransitCount++;
          }
        }

        if (inTransitCount < 3 || outTransitCount < 10) continue;

        const inTransitMean = inTransitSum / inTransitCount;
        const outTransitMean = outTransitSum / outTransitCount;
        const depth = outTransitMean - inTransitMean;

        // BLS power: proportional to depth squared times number of in-transit points
        const power =
          (depth * depth * inTransitCount * outTransitCount) /
          (inTransitCount + outTransitCount);

        if (power > bestPower && depth > 0) {
          bestPower = power;
          bestPeriod = period;
          bestPhase = phaseStart;
          bestDuration = transitDuration;
        }
      }
    }

    powers.push(bestPeriod === period ? bestPower : 0);
  }

  // Calculate transit parameters if a significant signal was found
  let transitParams: TransitParameters | null = null;

  if (bestPower > 0) {
    // Re-calculate depth and SNR for best period
    const phases = time.map((t) => ((t - time[0]) % bestPeriod) / bestPeriod);

    let inTransitFlux: number[] = [];
    let outTransitFlux: number[] = [];

    for (let i = 0; i < phases.length; i++) {
      const phase = phases[i];
      if (phase >= bestPhase && phase < bestPhase + bestDuration) {
        inTransitFlux.push(flux[i]);
      } else {
        outTransitFlux.push(flux[i]);
      }
    }

    if (inTransitFlux.length > 0 && outTransitFlux.length > 0) {
      const inMean = inTransitFlux.reduce((a, b) => a + b, 0) / inTransitFlux.length;
      const outMean = outTransitFlux.reduce((a, b) => a + b, 0) / outTransitFlux.length;
      const depth = (outMean - inMean) / outMean;
      const noise = std(outTransitFlux) / outMean;
      const snr = depth / (noise / Math.sqrt(inTransitFlux.length));

      // Calculate epoch (time of first transit)
      const epoch = time[0] + bestPhase * bestPeriod;

      transitParams = {
        period: bestPeriod,
        epoch: epoch,
        depth: depth,
        depth_ppm: depth * 1e6,
        duration: bestDuration * bestPeriod * 24, // Convert to hours
        snr: snr,
        power: bestPower,
      };
    }
  }

  // ── Hint-period fine search ────────────────────────────────────────────────
  // For targets with a known/suspected period (e.g. KNOWN_TARGETS), run an
  // extra high-resolution sweep at the exact period.  This handles long-period
  // planets whose transit-duration fraction is too small for the general BLS
  // (e.g. TOI-1231.01: 0.078d / 24.25d = 0.0032 < minTransitDuration 0.01).
  if (hintPeriod !== undefined && hintPeriod > 0) {
    const hPhases = time.map(t => ((t - time[0]) % hintPeriod) / hintPeriod);
    const hNorm   = normFlux; // already computed above

    // 12 duration steps from 0.001 to 0.2 of the hint period
    const hDurSteps = 12;
    // 2000 phase steps — fine enough for any transit fraction ≥ 0.001
    const hPhaseSteps = 2000;

    for (let d = 0; d < hDurSteps; d++) {
      const td = 0.001 + (0.20 - 0.001) * d / hDurSteps;

      for (let p = 0; p < hPhaseSteps; p++) {
        const phaseStart = p / hPhaseSteps;

        let iti = 0, itc = 0, oti = 0, otc = 0;
        for (let i = 0; i < hPhases.length; i++) {
          const ph = hPhases[i];
          const wp = ph < phaseStart ? ph + 1 : ph;
          if (wp >= phaseStart && wp < phaseStart + td) {
            iti += hNorm[i]; itc++;
          } else {
            oti += hNorm[i]; otc++;
          }
        }
        if (itc < 3 || otc < 10) continue;

        const dep = oti / otc - iti / itc;
        const pw  = dep > 0
          ? (dep * dep * itc * otc) / (itc + otc)
          : 0;

        if (pw > bestPower) {
          bestPower    = pw;
          bestPeriod   = hintPeriod;
          bestPhase    = phaseStart;
          bestDuration = td;
        }
      }
    }

    // Re-derive transitParams if hint search beat general BLS
    if (bestPeriod === hintPeriod && bestPower > 0) {
      const hPh2 = time.map(t => ((t - time[0]) % hintPeriod) / hintPeriod);
      const inF: number[] = [], outF: number[] = [];
      for (let i = 0; i < hPh2.length; i++) {
        if (hPh2[i] >= bestPhase && hPh2[i] < bestPhase + bestDuration) {
          inF.push(flux[i]);
        } else {
          outF.push(flux[i]);
        }
      }
      if (inF.length > 0 && outF.length > 0) {
        const inMean  = inF.reduce((a,b)=>a+b,0) / inF.length;
        const outMean = outF.reduce((a,b)=>a+b,0) / outF.length;
        const dep     = (outMean - inMean) / outMean;
        const nz      = std(outF) / outMean;
        const snr     = dep / (nz / Math.sqrt(inF.length));
        transitParams = {
          period:    hintPeriod,
          epoch:     time[0] + bestPhase * hintPeriod,
          depth:     dep,
          depth_ppm: dep * 1e6,
          duration:  bestDuration * hintPeriod * 24,
          snr,
          power:     bestPower,
        };
      }
    }
  }
  // ──────────────────────────────────────────────────────────────────────────

  return {
    bestPeriod,
    bestPower,
    periods,
    powers,
    transitParams,
  };
}

/**
 * Run vetting tests on detected transit signal
 */
export function runVettingTests(
  time: number[],
  flux: number[],
  params: TransitParameters
): VettingResult {
  const tests: VettingResult = {
    disposition: 'INCONCLUSIVE',
    confidence: 0,
    odd_even: { flag: 'WARNING', message: '', confidence: 0.5 },
    v_shape: { flag: 'WARNING', message: '', confidence: 0.5 },
    secondary_eclipse: { flag: 'WARNING', message: '', confidence: 0.5 },
  };

  // Phase fold the data
  const phases = time.map(
    (t) => ((t - params.epoch) % params.period) / params.period
  );

  // Normalize phases to [-0.5, 0.5]
  const normPhases = phases.map((p) => (p > 0.5 ? p - 1 : p));

  // 1. Odd/Even test - check if odd and even transits have same depth
  const oddTransits: number[] = [];
  const evenTransits: number[] = [];
  const transitPhase = params.duration / (params.period * 24);

  for (let i = 0; i < phases.length; i++) {
    if (Math.abs(normPhases[i]) < transitPhase) {
      const transitNumber = Math.floor(
        (time[i] - params.epoch) / params.period
      );
      if (transitNumber % 2 === 0) {
        evenTransits.push(flux[i]);
      } else {
        oddTransits.push(flux[i]);
      }
    }
  }

  if (oddTransits.length > 5 && evenTransits.length > 5) {
    const oddDepth = 1 - median(oddTransits);
    const evenDepth = 1 - median(evenTransits);
    const depthDiff = Math.abs(oddDepth - evenDepth);
    const avgDepth = (oddDepth + evenDepth) / 2;
    const significance = depthDiff / avgDepth;

    if (significance < 0.1) {
      tests.odd_even = {
        flag: 'PASS',
        message: `Odd/even depth difference: ${(significance * 100).toFixed(1)}%`,
        confidence: 0.95 - significance * 2,
      };
    } else if (significance < 0.3) {
      tests.odd_even = {
        flag: 'WARNING',
        message: `Odd/even depth difference: ${(significance * 100).toFixed(1)}% (marginal)`,
        confidence: 0.7 - significance,
      };
    } else {
      tests.odd_even = {
        flag: 'FAIL',
        message: `Significant odd/even depth difference: ${(significance * 100).toFixed(1)}% - possible eclipsing binary`,
        confidence: 0.3,
      };
    }
  } else {
    tests.odd_even = {
      flag: 'WARNING',
      message: 'Insufficient transits for odd/even test',
      confidence: 0.5,
    };
  }

  // 2. V-shape test - check transit shape
  const inTransitFlux: { phase: number; flux: number }[] = [];
  for (let i = 0; i < phases.length; i++) {
    if (Math.abs(normPhases[i]) < transitPhase * 1.5) {
      inTransitFlux.push({ phase: normPhases[i], flux: flux[i] });
    }
  }

  if (inTransitFlux.length > 20) {
    // Sort by phase
    inTransitFlux.sort((a, b) => a.phase - b.phase);

    // Check for flat bottom (planetary) vs V-shape (eclipsing binary)
    const bottomPhase = transitPhase * 0.3;
    const bottomFlux = inTransitFlux
      .filter((p) => Math.abs(p.phase) < bottomPhase)
      .map((p) => p.flux);
    const ingressFlux = inTransitFlux
      .filter((p) => p.phase < -bottomPhase && p.phase > -transitPhase)
      .map((p) => p.flux);

    if (bottomFlux.length > 5 && ingressFlux.length > 5) {
      const bottomStd = std(bottomFlux);
      const ingressStd = std(ingressFlux);
      const flatness = bottomStd / ingressStd;

      if (flatness < 0.5) {
        tests.v_shape = {
          flag: 'PASS',
          message: 'Transit shape consistent with planetary (flat bottom)',
          confidence: 0.9,
        };
      } else if (flatness < 1.0) {
        tests.v_shape = {
          flag: 'WARNING',
          message: 'Transit shape slightly V-shaped',
          confidence: 0.6,
        };
      } else {
        tests.v_shape = {
          flag: 'FAIL',
          message: 'V-shaped transit - possible eclipsing binary or grazing transit',
          confidence: 0.3,
        };
      }
    }
  }

  // 3. Secondary eclipse test - check for eclipse at phase 0.5
  const secondaryPhaseRange = 0.05;
  const secondaryFlux = flux.filter(
    (_, i) => Math.abs(normPhases[i] - 0.5) < secondaryPhaseRange ||
              Math.abs(normPhases[i] + 0.5) < secondaryPhaseRange
  );
  const outOfTransitFlux = flux.filter(
    (_, i) =>
      Math.abs(normPhases[i]) > transitPhase * 2 &&
      Math.abs(Math.abs(normPhases[i]) - 0.5) > secondaryPhaseRange
  );

  if (secondaryFlux.length > 10 && outOfTransitFlux.length > 50) {
    const secondaryMedian = median(secondaryFlux);
    const ootMedian = median(outOfTransitFlux);
    const ootMad = mad(outOfTransitFlux) * 1.4826;

    const secondaryDepth = (ootMedian - secondaryMedian) / ootMedian;
    const significance = secondaryDepth / (ootMad / Math.sqrt(secondaryFlux.length));

    if (significance < 3) {
      tests.secondary_eclipse = {
        flag: 'PASS',
        message: 'No significant secondary eclipse detected',
        confidence: 0.9,
      };
    } else if (significance < 5) {
      tests.secondary_eclipse = {
        flag: 'WARNING',
        message: `Possible weak secondary eclipse (${significance.toFixed(1)}σ)`,
        confidence: 0.6,
      };
    } else {
      tests.secondary_eclipse = {
        flag: 'FAIL',
        message: `Secondary eclipse detected (${significance.toFixed(1)}σ) - likely eclipsing binary`,
        confidence: 0.2,
      };
    }
  }

  // Calculate overall disposition
  const passCount = [tests.odd_even, tests.v_shape, tests.secondary_eclipse].filter(
    (t) => t.flag === 'PASS'
  ).length;
  const failCount = [tests.odd_even, tests.v_shape, tests.secondary_eclipse].filter(
    (t) => t.flag === 'FAIL'
  ).length;

  if (failCount >= 2) {
    tests.disposition = 'LIKELY_FALSE_POSITIVE';
    tests.confidence = 0.3;
  } else if (passCount >= 2 && failCount === 0) {
    tests.disposition = 'PLANET_CANDIDATE';
    tests.confidence =
      (tests.odd_even.confidence +
        tests.v_shape.confidence +
        tests.secondary_eclipse.confidence) /
      3;
  } else {
    tests.disposition = 'INCONCLUSIVE';
    tests.confidence = 0.5;
  }

  return tests;
}

/**
 * Compute a phase-folded, binned light curve for display
 * Returns 100 {phase, flux} points centered on transit (phase=0)
 */
function computeFoldedCurve(
  time: number[],
  flux: number[],
  period: number,
  epoch: number,
  nBins: number = 100
): { phase: number; flux: number }[] {
  // Phase-fold: map each time to [-0.5, 0.5)
  const phases = time.map(t => {
    let p = ((t - epoch) % period) / period;
    if (p > 0.5) p -= 1;
    if (p < -0.5) p += 1;
    return p;
  });

  // Bin the folded data
  const bins: number[][] = Array.from({ length: nBins }, () => []);
  for (let i = 0; i < phases.length; i++) {
    const binIdx = Math.min(nBins - 1, Math.floor((phases[i] + 0.5) * nBins));
    if (binIdx >= 0) bins[binIdx].push(flux[i]);
  }

  return bins.map((bin, idx) => ({
    phase: (idx + 0.5) / nBins - 0.5,
    flux: bin.length > 0 ? bin.reduce((a, b) => a + b, 0) / bin.length : 1.0,
  }));
}

/**
 * Main detection function
 */
export async function runTransitDetection(
  lightCurve: LightCurveData,
  ticInfo: TICInfo | null,
  hintPeriod?: number     // Known period to test with fine-grid search
): Promise<DetectionResult> {
  const startTime = Date.now();

  // Filter out bad quality data
  const goodMask = lightCurve.quality.map((q) => q === 0);
  let time = lightCurve.time.filter((_, i) => goodMask[i]);
  let flux = lightCurve.flux.filter((_, i) => goodMask[i]);

  // Remove NaN values
  const validMask = flux.map((f) => !isNaN(f) && isFinite(f));
  time = time.filter((_, i) => validMask[i]);
  flux = flux.filter((_, i) => validMask[i]);

  if (time.length < 100) {
    return {
      detection: false,
      confidence: 0,
      period_days: null,
      depth_ppm: null,
      duration_hours: null,
      epoch_btjd: null,
      snr: null,
      vetting: null,
      sectors_used: lightCurve.sectors,
      processing_time_seconds: (Date.now() - startTime) / 1000,
    };
  }

  // Sigma clip at 20σ — removes cosmic-ray spikes only.
  // Lower values (e.g. 5σ) would clip deep transit signals (8000–9000 ppm)
  // in low-noise data, causing false "no detection" results.
  const clipped = sigmaClip(time, flux, 20);
  time = clipped.time;
  flux = clipped.flux;

  // Detrend the light curve
  flux = detrendLightCurve(time, flux);

  // Downsample to ≤3000 points before BLS.
  // BLS is O(N_periods × N_phases × N_data); this cuts ~6–7s → ~0.3s.
  const ds = downsampleTimeSeries(time, flux, 3000);
  const blsTime = ds.time;
  const blsFlux = ds.flux;

  // Run BLS (pass hint period for targeted fine search)
  const blsResult = runBLS(blsTime, blsFlux, { hintPeriod });

  // Check if we have a significant detection
  const detectionThreshold = 0.0001; // Minimum BLS power
  const snrThreshold = 7.0; // Minimum SNR

  if (
    !blsResult.transitParams ||
    blsResult.bestPower < detectionThreshold ||
    blsResult.transitParams.snr < snrThreshold
  ) {
    return {
      detection: false,
      confidence: blsResult.transitParams
        ? Math.min(0.4, blsResult.transitParams.snr / 20)
        : 0.1,
      period_days: null,
      depth_ppm: null,
      duration_hours: null,
      epoch_btjd: null,
      snr: blsResult.transitParams?.snr || null,
      vetting: null,
      sectors_used: lightCurve.sectors,
      processing_time_seconds: (Date.now() - startTime) / 1000,
    };
  }

  // Run vetting tests (use full-res arrays for accuracy)
  const vetting = runVettingTests(time, flux, blsResult.transitParams);

  // Calculate final confidence
  let confidence = Math.min(
    0.99,
    0.5 + (blsResult.transitParams.snr - snrThreshold) / 30
  );

  // Adjust confidence based on vetting
  if (vetting.disposition === 'LIKELY_FALSE_POSITIVE') {
    confidence *= 0.3;
  } else if (vetting.disposition === 'INCONCLUSIVE') {
    confidence *= 0.7;
  }

  // Compute phase-folded curve for display (full-res, single pass)
  const folded_curve = computeFoldedCurve(
    time, flux,
    blsResult.transitParams.period,
    blsResult.transitParams.epoch
  );

  return {
    detection: true,
    confidence: Number(confidence.toFixed(3)),
    period_days: Number(blsResult.transitParams.period.toFixed(4)),
    depth_ppm: Number(blsResult.transitParams.depth_ppm.toFixed(0)),
    duration_hours: Number(blsResult.transitParams.duration.toFixed(2)),
    epoch_btjd: Number(blsResult.transitParams.epoch.toFixed(4)),
    snr: Number(blsResult.transitParams.snr.toFixed(1)),
    vetting,
    sectors_used: lightCurve.sectors,
    processing_time_seconds: Number(((Date.now() - startTime) / 1000).toFixed(2)),
    folded_curve,
  };
}
