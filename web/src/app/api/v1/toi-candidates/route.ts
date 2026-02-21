import { NextResponse } from 'next/server';

export const revalidate = 3600; // cache 1 hour

export interface TOICandidate {
  tic_id: string;
  toi: string;
  period_days: number;
  depth_ppm: number;
  duration_hours: number;
  disposition: 'PC' | 'APC' | 'CP' | 'FP' | 'KP';
  planet_radius_earth: number | null;
  star_tmag: number | null;
}

const NASA_TAP =
  'https://exoplanetarchive.ipac.caltech.edu/TAP/sync';

/**
 * Fetch live TOI candidates from NASA Exoplanet Archive.
 * Returns PCs (Planet Candidates) and CPs (Confirmed Planets) with enough
 * transit parameters to run BLS analysis.
 */
export async function GET() {
  try {
    // NASA TAP uses Oracle/ADQL: TOP not LIMIT, column is tid (TIC ID) and pl_trandurh
    const query = `
      SELECT TOP 200 tid, toi, pl_orbper, pl_trandep, pl_trandurh,
             tfopwg_disp, pl_rade, st_tmag
      FROM toi
      WHERE tfopwg_disp IN ('PC','APC','CP','KP')
        AND pl_orbper IS NOT NULL
        AND pl_trandep IS NOT NULL
        AND pl_trandurh IS NOT NULL
        AND pl_orbper > 0.3
        AND pl_orbper < 50
        AND pl_trandep > 100
      ORDER BY toi ASC
    `.replace(/\s+/g, ' ').trim();

    const url = `${NASA_TAP}?query=${encodeURIComponent(query)}&format=json`;
    const res = await fetch(url, {
      next: { revalidate: 3600 },
      headers: { 'User-Agent': 'LARUN-Exoplanet-Platform/2.0' },
    });

    if (!res.ok) {
      throw new Error(`NASA TAP returned ${res.status}`);
    }

    const raw: Record<string, string | number | null>[] = await res.json();

    const candidates: TOICandidate[] = raw
      .filter(r => r.tid != null)
      .map(r => ({
        tic_id: String(r.tid),
        toi: String(r.toi),
        period_days: Number(r.pl_orbper),
        depth_ppm: Number(r.pl_trandep),           // already in ppm from archive
        duration_hours: Number(r.pl_trandurh),     // NASA column is pl_trandurh (hours)
        disposition: (r.tfopwg_disp as TOICandidate['disposition']) ?? 'PC',
        planet_radius_earth: r.pl_rade != null ? Number(r.pl_rade) : null,
        star_tmag: r.st_tmag != null ? Number(r.st_tmag) : null,
      }));

    return NextResponse.json({ candidates, total: candidates.length, source: 'NASA Exoplanet Archive' });
  } catch (err) {
    console.error('TOI fetch error:', err);
    return NextResponse.json(
      { error: 'Failed to fetch TOI catalog', details: String(err) },
      { status: 502 },
    );
  }
}
