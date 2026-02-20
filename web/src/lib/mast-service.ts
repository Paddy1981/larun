/**
 * MAST (Mikulski Archive for Space Telescopes) Service
 *
 * Fetches real TESS light curve data from NASA's MAST archive.
 * Documentation: https://mast.stsci.edu/api/v0/
 */

export interface LightCurveData {
  time: number[];  // BJD timestamps
  flux: number[];  // Normalized flux values
  flux_err: number[];  // Flux errors
  quality: number[];  // Quality flags
  sectors: number[];  // TESS sectors
}

export interface TICInfo {
  tic_id: string;
  ra: number;
  dec: number;
  tmag: number;  // TESS magnitude
  teff: number;  // Effective temperature
  radius: number;  // Stellar radius (solar radii)
  mass: number;  // Stellar mass (solar masses)
  logg: number;  // Surface gravity
}

const MAST_API_URL = 'https://mast.stsci.edu/api/v0';
const MAST_PORTAL_URL = 'https://mast.stsci.edu/portal/Download/file';

/** Fetch with a hard timeout (aborts cleanly; avoids 60s Vercel hard kill) */
async function fetchWithTimeout(
  url: string,
  init: RequestInit = {},
  timeoutMs = 12_000
): Promise<Response> {
  const ctrl = new AbortController();
  const id   = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: ctrl.signal });
  } finally {
    clearTimeout(id);
  }
}

// ---------------------------------------------------------------------------
// Minimal FITS binary table parser (Node.js-compatible, no browser APIs)
// Handles TESS LC FITS files: PRIMARY + LIGHTCURVE binary table HDU.
// fitsjs (installed) is browser-only; this parser uses DataView + ArrayBuffer.
// ---------------------------------------------------------------------------
const FITS_BLOCK = 2880;
const CARD_LEN   = 80;

function readFITSHeader(
  buffer: ArrayBuffer,
  byteOffset: number
): { header: Map<string, string>; nextOffset: number } {
  const bytes = new Uint8Array(buffer);
  const header = new Map<string, string>();
  let pos = byteOffset;
  let ended = false;

  while (!ended && pos + FITS_BLOCK <= bytes.length) {
    for (let card = 0; card < 36 && !ended; card++) {
      const start = pos + card * CARD_LEN;
      // Decode the 80-byte card as ASCII
      let line = '';
      for (let b = 0; b < CARD_LEN; b++) line += String.fromCharCode(bytes[start + b]);

      const keyword = line.slice(0, 8).trimEnd();
      if (keyword === 'END') { ended = true; break; }

      // Value indicators: standard '= ' at position 8-9
      if (line[8] === '=' ) {
        const raw = line.slice(10);
        const slash = raw.indexOf('/');
        const val = (slash >= 0 ? raw.slice(0, slash) : raw)
          .trim()
          .replace(/^'(.*?)'.*$/, '$1')  // strip FITS string quotes
          .trim();
        header.set(keyword, val);
      }
    }
    pos += FITS_BLOCK;
  }

  return { header, nextOffset: pos };
}

/** Parse TFORM like '1D', 'E', '28D' → bytes per element and type char */
function parseTFORM(tform: string): { type: string; bytes: number } {
  const m = tform.trim().match(/^(\d*)([DEJIKLAB])(\(.*\))?$/i);
  if (!m) return { type: 'E', bytes: 4 };
  const count = m[1] ? parseInt(m[1]) : 1;
  const type  = m[2].toUpperCase();
  const sizeOf: Record<string, number> = { D: 8, E: 4, J: 4, I: 2, K: 8, L: 1, A: 1, B: 1 };
  return { type, bytes: (sizeOf[type] ?? 4) * count };
}

function parseFITSLightCurve(buffer: ArrayBuffer): {
  time: number[];
  flux: number[];
  flux_err: number[];
  quality: number[];
} {
  // --- PRIMARY HDU ---
  const { header: primaryHeader, nextOffset: afterPrimaryHeader } = readFITSHeader(buffer, 0);
  let offset = afterPrimaryHeader;

  // Skip primary data block if present (NAXIS > 0)
  const naxisPrimary = parseInt(primaryHeader.get('NAXIS') ?? '0');
  if (naxisPrimary > 0) {
    let dataBytes = Math.abs(parseInt(primaryHeader.get('BITPIX') ?? '8')) / 8;
    for (let i = 1; i <= naxisPrimary; i++) {
      dataBytes *= parseInt(primaryHeader.get(`NAXIS${i}`) ?? '0');
    }
    offset += Math.ceil(dataBytes / FITS_BLOCK) * FITS_BLOCK;
  }

  // --- LIGHTCURVE extension HDU ---
  const { header, nextOffset: afterHeader } = readFITSHeader(buffer, offset);
  offset = afterHeader;

  const nRows   = parseInt(header.get('NAXIS2') ?? '0');
  const nFields = parseInt(header.get('TFIELDS') ?? '0');
  const rowSize = parseInt(header.get('NAXIS1') ?? '0');

  if (nRows === 0 || nFields === 0 || rowSize === 0) {
    throw new Error('FITS binary table appears empty');
  }

  // Build column layout: name → {type, bytes, colOffset}
  const cols: { name: string; type: string; bytes: number; colOffset: number }[] = [];
  let colOff = 0;
  for (let i = 1; i <= nFields; i++) {
    const name   = (header.get(`TTYPE${i}`) ?? '').trim();
    const tform  = header.get(`TFORM${i}`) ?? 'E';
    const { type, bytes } = parseTFORM(tform);
    cols.push({ name, type, bytes, colOffset: colOff });
    colOff += bytes;
  }

  const col = (name: string) => cols.find(c => c.name === name);
  const timeCol    = col('TIME');
  const fluxCol    = col('PDCSAP_FLUX') ?? col('SAP_FLUX');
  const fluxErrCol = col('PDCSAP_FLUX_ERR') ?? col('SAP_FLUX_ERR');
  const qualCol    = col('QUALITY');

  if (!timeCol || !fluxCol) throw new Error('TIME or FLUX column not found in FITS table');

  const view = new DataView(buffer);
  const time: number[] = [], flux: number[] = [], flux_err: number[] = [], quality: number[] = [];

  for (let row = 0; row < nRows; row++) {
    const base = offset + row * rowSize;
    // FITS is big-endian (littleEndian = false)
    const t  = view.getFloat64(base + timeCol.colOffset, false);
    const f  = view.getFloat32(base + fluxCol.colOffset, false);
    const fe = fluxErrCol ? view.getFloat32(base + fluxErrCol.colOffset, false) : 0.001;
    const q  = qualCol    ? view.getInt32(base + qualCol.colOffset,    false) : 0;

    if (!isFinite(t) || !isFinite(f) || isNaN(t) || isNaN(f)) continue;
    time.push(t);
    flux.push(f);
    flux_err.push(!isFinite(fe) || isNaN(fe) ? 0.001 : Math.abs(fe));
    quality.push(q);
  }

  if (time.length === 0) throw new Error('No valid data rows in FITS table');

  // Normalize flux to median ≈ 1.0
  const sorted = [...flux].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  const normFlux    = flux.map(f => f / med);
  const normFluxErr = flux_err.map(e => e / med);

  return { time, flux: normFlux, flux_err: normFluxErr, quality };
}

/**
 * Query MAST for observations of a TIC target
 */
async function queryMAST(ticId: string): Promise<Record<string, unknown>[]> {
  const request = {
    service: 'Mast.Caom.Filtered',
    format: 'json',
    params: {
      columns: 'obsid,obs_collection,dataproduct_type,obs_id,target_name,s_ra,s_dec,t_min,t_max,sequence_number',
      filters: [
        { paramName: 'obs_collection', values: ['TESS'] },
        { paramName: 'dataproduct_type', values: ['timeseries'] },
        { paramName: 'target_name', values: [`TIC ${ticId}`] },
      ],
    },
  };

  const response = await fetchWithTimeout(`${MAST_API_URL}/invoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`MAST query failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.data || [];
}

/**
 * Get data products for an observation
 */
async function getDataProducts(obsId: string): Promise<Record<string, unknown>[]> {
  const request = {
    service: 'Mast.Caom.Products',
    format: 'json',
    params: { obsid: obsId },
  };

  const response = await fetchWithTimeout(`${MAST_API_URL}/invoke`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`MAST products query failed: ${response.statusText}`);
  }

  const data = await response.json();
  return data.data || [];
}

/**
 * Download and parse a TESS light curve file.
 * Tries our custom FITS binary table parser first (handles .lc.fits files),
 * falls back to JSON parsing for any other format.
 */
async function downloadLightCurve(dataUrl: string): Promise<{
  time: number[];
  flux: number[];
  flux_err: number[];
  quality: number[];
}> {
  const response = await fetchWithTimeout(dataUrl, {}, 20_000); // FITS can be large

  if (!response.ok) {
    throw new Error(`Failed to download light curve: ${response.statusText}`);
  }

  const contentType = response.headers.get('content-type') ?? '';

  // Try JSON first (fast path)
  if (contentType.includes('json')) {
    const data = await response.json();
    return parseLightCurveJSON(data);
  }

  // FITS binary format — use our Node.js-compatible parser
  // (fitsjs is browser-only; parseFITSLightCurve uses DataView/ArrayBuffer)
  const buffer = await response.arrayBuffer();
  const magic = new Uint8Array(buffer, 0, 9);
  const header = String.fromCharCode(...magic);
  if (header.startsWith('SIMPLE  ') || header.startsWith('SIMPLE =')) {
    return parseFITSLightCurve(buffer);
  }

  // Last resort: treat as JSON text
  const text = new TextDecoder().decode(buffer);
  const data = JSON.parse(text);
  return parseLightCurveJSON(data);
}

/**
 * Parse light curve data from JSON response
 */
function parseLightCurveJSON(data: Record<string, unknown>): {
  time: number[];
  flux: number[];
  flux_err: number[];
  quality: number[];
} {
  // Handle different JSON formats from MAST
  if (Array.isArray(data)) {
    const time: number[] = [];
    const flux: number[] = [];
    const flux_err: number[] = [];
    const quality: number[] = [];

    for (const row of data) {
      const r = row as Record<string, number>;
      if (r.time !== undefined && r.flux !== undefined) {
        time.push(r.time);
        flux.push(r.flux);
        flux_err.push(r.flux_err || 0.001);
        quality.push(r.quality || 0);
      }
    }

    return { time, flux, flux_err, quality };
  }

  throw new Error('Unexpected light curve data format');
}

/**
 * Fetch TIC information from MAST
 */
export async function fetchTICInfo(ticId: string): Promise<TICInfo | null> {
  try {
    const request = {
      service: 'Mast.Catalogs.Filtered.Tic',
      format: 'json',
      params: {
        columns: 'ID,ra,dec,Tmag,Teff,rad,mass,logg',
        filters: [
          { paramName: 'ID', values: [ticId] },
        ],
      },
    };

    const response = await fetchWithTimeout(`${MAST_API_URL}/invoke`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });

    if (!response.ok) return null;

    const data = await response.json();

    if (data.data && data.data.length > 0) {
      const row = data.data[0];
      return {
        tic_id: ticId,
        ra: row.ra || 0,
        dec: row.dec || 0,
        tmag: row.Tmag || 0,
        teff: row.Teff || 5778, // Sun's effective temperature as default
        radius: row.rad || 1,
        mass: row.mass || 1,
        logg: row.logg || 4.44,
      };
    }

    return null;
  } catch (error) {
    console.error('Error fetching TIC info:', error);
    return null;
  }
}

// Known transit parameters for confirmed/candidate targets (exported for hint-period detection)
export const KNOWN_TARGETS: Record<string, { period: number; depth: number; duration: number }> = {
  '470710327': { period: 14.61, depth: 0.0052, duration: 0.083 }, // TOI-1338 b
  '307210830': { period: 37.42, depth: 0.0021, duration: 0.095 }, // TOI-700 d
  '441462736': { period: 0.766, depth: 0.0089, duration: 0.042 }, // TOI-849 b
  '141527579': { period: 0.449, depth: 0.0018, duration: 0.031 }, // TOI-561 b
  '231702397': { period: 24.25, depth: 0.0045, duration: 0.078 }, // TOI-1231.01
  '396740648': { period: 7.85,  depth: 0.0038, duration: 0.065 }, // TOI-2136.01
  '267263253': { period: 11.06, depth: 0.0052, duration: 0.072 }, // TOI-1452.01
  '150428135': { period: 3.13,  depth: 0.0028, duration: 0.051 }, // TOI-1695.01
  '219195044': { period: 18.85, depth: 0.0041, duration: 0.069 }, // TOI-1759.01
  '261136679': { period: 3.69,  depth: 0.0019, duration: 0.044 }, // TOI-175 b
};

/**
 * Generate synthetic light curve for testing
 * Used when real data is not available
 */
export function generateSyntheticLightCurve(
  ticId: string,
  options: {
    hasTransit?: boolean;
    period?: number;
    depth?: number;
    duration?: number;
    noise?: number;
  } = {}
): LightCurveData {
  // Use seeded deterministic random so same TIC always gives same result
  let seedVal = 0;
  for (let i = 0; i < ticId.length; i++) seedVal += ticId.charCodeAt(i) * (i + 1);
  const seededRand = (() => {
    let s = seedVal;
    return () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  })();

  const known = KNOWN_TARGETS[ticId];
  const {
    hasTransit = known ? true : seededRand() > 0.4,
    period = known?.period ?? (1 + seededRand() * 25),
    depth = known?.depth ?? (0.001 + seededRand() * 0.008),
    duration = known?.duration ?? (0.05 + seededRand() * 0.1),
    noise = 0.0004 + seededRand() * 0.0008,
  } = options;

  // Generate ~27 days of data (one TESS sector) at 15-minute cadence.
  // 15-min binned cadence (vs 2-min native) reduces BLS computation ~50x
  // while still resolving transits ≥45 min (all known targets qualify).
  const cadenceMin = 15;
  const nPoints = Math.floor(27 * 24 * 60 / cadenceMin); // 2592 points
  const time: number[] = [];
  const flux: number[] = [];
  const flux_err: number[] = [];
  const quality: number[] = [];

  const t0 = 1325.0; // Reference epoch (BTJD)

  for (let i = 0; i < nPoints; i++) {
    const t = t0 + (i * cadenceMin) / (24 * 60); // 15-min cadence
    time.push(t);

    // Base flux with Gaussian noise
    let f = 1.0 + (seededRand() - 0.5) * noise * 2;

    // Add transit if applicable
    if (hasTransit) {
      const phase = ((t - t0) % period) / period;
      const transitPhase = duration / period;

      // Simple box transit model
      if (phase < transitPhase || phase > (1 - transitPhase / 2)) {
        // In transit
        const ingressEgress = transitPhase * 0.1;
        if (phase < ingressEgress) {
          // Ingress
          f -= depth * (phase / ingressEgress);
        } else if (phase < transitPhase - ingressEgress) {
          // Full transit
          f -= depth;
        } else if (phase < transitPhase) {
          // Egress
          f -= depth * ((transitPhase - phase) / ingressEgress);
        }
      }
    }

    flux.push(f);
    flux_err.push(noise);
    quality.push(0);
  }

  return {
    time,
    flux,
    flux_err,
    quality,
    sectors: [1],
  };
}

/**
 * Fetch light curve data for a TIC target
 * Falls back to synthetic data if real data is unavailable
 */
export async function fetchLightCurve(ticId: string): Promise<LightCurveData> {
  try {
    // Query MAST for observations
    const observations = await queryMAST(ticId);

    if (observations.length === 0) {
      console.log(`No TESS observations found for TIC ${ticId}, using synthetic data`);
      return generateSyntheticLightCurve(ticId);
    }

    // Get the most recent observation
    const latestObs = observations[0] as { obsid: string; sequence_number: number };
    const obsId = latestObs.obsid;

    // Get data products
    const products = await getDataProducts(obsId);

    // Find the light curve product (prefer PDCSAP flux)
    const lcProduct = products.find((p) => {
      const prod = p as { productSubGroupDescription?: string; dataURI?: string };
      return prod.productSubGroupDescription?.includes('LC') &&
             prod.dataURI?.includes('lc.fits');
    }) as { dataURI?: string } | undefined;

    if (!lcProduct || !lcProduct.dataURI) {
      console.log(`No light curve product found for TIC ${ticId}, using synthetic data`);
      return generateSyntheticLightCurve(ticId);
    }

    // Download and parse the light curve
    const lcData = await downloadLightCurve(
      `${MAST_PORTAL_URL}?uri=${encodeURIComponent(lcProduct.dataURI)}`
    );

    return {
      ...lcData,
      sectors: [latestObs.sequence_number || 1],
    };
  } catch (error) {
    console.error(`Error fetching light curve for TIC ${ticId}:`, error);
    // Fall back to synthetic data
    return generateSyntheticLightCurve(ticId);
  }
}
