"""
NASA Data Pipeline
==================
Handles data ingestion from various NASA archives including MAST, 
Exoplanet Archive, and IRSA.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations, Catalogs
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import requests
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SpectralData:
    """Container for spectral data from NASA sources."""
    wavelength: np.ndarray
    flux: np.ndarray
    flux_error: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    object_id: str = ""
    observation_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "wavelength": self.wavelength.tolist(),
            "flux": self.flux.tolist(),
            "flux_error": self.flux_error.tolist() if self.flux_error is not None else None,
            "time": self.time.tolist() if self.time is not None else None,
            "metadata": self.metadata,
            "source": self.source,
            "object_id": self.object_id,
            "observation_date": self.observation_date.isoformat() if self.observation_date else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpectralData":
        return cls(
            wavelength=np.array(data["wavelength"]),
            flux=np.array(data["flux"]),
            flux_error=np.array(data["flux_error"]) if data.get("flux_error") else None,
            time=np.array(data["time"]) if data.get("time") else None,
            metadata=data.get("metadata", {}),
            source=data.get("source", "unknown"),
            object_id=data.get("object_id", ""),
            observation_date=datetime.fromisoformat(data["observation_date"]) if data.get("observation_date") else None
        )


class NASADataPipeline:
    """
    Main pipeline for ingesting data from NASA archives.
    
    Supports:
    - MAST (Mikulski Archive for Space Telescopes)
    - NASA Exoplanet Archive
    - IRSA (Infrared Science Archive)
    """
    
    def __init__(self, config: Dict[str, Any], cache_dir: str = "data/raw"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize API tokens if available
        self.mast_token = os.getenv("MAST_API_TOKEN")
        if self.mast_token:
            Observations.login(token=self.mast_token)
    
    async def fetch_kepler_lightcurve(
        self, 
        target_name: str, 
        quarter: Optional[int] = None
    ) -> List[SpectralData]:
        """
        Fetch Kepler light curve data for a target.
        
        Args:
            target_name: Target identifier (e.g., "Kepler-186")
            quarter: Specific quarter to fetch (None for all)
            
        Returns:
            List of SpectralData objects
        """
        logger.info(f"Fetching Kepler data for {target_name}")
        
        try:
            from lightkurve import search_lightcurve
            
            # Search for light curves
            search_result = search_lightcurve(
                target_name, 
                mission="Kepler",
                quarter=quarter
            )
            
            if len(search_result) == 0:
                logger.warning(f"No Kepler data found for {target_name}")
                return []
            
            results = []
            for i, lc_file in enumerate(search_result):
                try:
                    lc = lc_file.download()
                    
                    spectral_data = SpectralData(
                        wavelength=np.arange(len(lc.flux)),  # Using index as pseudo-wavelength
                        flux=lc.flux.value,
                        flux_error=lc.flux_err.value if hasattr(lc, 'flux_err') else None,
                        time=lc.time.value,
                        metadata={
                            "mission": "Kepler",
                            "quarter": lc_file.mission[0] if hasattr(lc_file, 'mission') else None,
                            "exptime": float(lc_file.exptime[0].value) if hasattr(lc_file, 'exptime') else None
                        },
                        source="MAST/Kepler",
                        object_id=target_name,
                        observation_date=datetime.now()  # Would be from actual observation
                    )
                    results.append(spectral_data)
                    
                except Exception as e:
                    logger.error(f"Error downloading light curve {i}: {e}")
                    continue
            
            return results
            
        except ImportError:
            logger.error("lightkurve not installed. Using direct MAST query.")
            return await self._fetch_mast_direct(target_name, "Kepler")
    
    async def fetch_tess_lightcurve(
        self, 
        target_name: str,
        sector: Optional[int] = None
    ) -> List[SpectralData]:
        """Fetch TESS light curve data for a target."""
        logger.info(f"Fetching TESS data for {target_name}")
        
        try:
            from lightkurve import search_lightcurve
            
            search_result = search_lightcurve(
                target_name,
                mission="TESS",
                sector=sector
            )
            
            if len(search_result) == 0:
                logger.warning(f"No TESS data found for {target_name}")
                return []
            
            results = []
            for lc_file in search_result:
                try:
                    lc = lc_file.download()
                    
                    spectral_data = SpectralData(
                        wavelength=np.arange(len(lc.flux)),
                        flux=lc.flux.value,
                        flux_error=lc.flux_err.value if hasattr(lc, 'flux_err') else None,
                        time=lc.time.value,
                        metadata={
                            "mission": "TESS",
                            "sector": sector
                        },
                        source="MAST/TESS",
                        object_id=target_name
                    )
                    results.append(spectral_data)
                    
                except Exception as e:
                    logger.error(f"Error downloading TESS light curve: {e}")
                    continue
            
            return results
            
        except ImportError:
            return await self._fetch_mast_direct(target_name, "TESS")
    
    async def _fetch_mast_direct(
        self, 
        target_name: str, 
        mission: str
    ) -> List[SpectralData]:
        """Direct MAST API query when lightkurve is not available."""
        
        obs_table = Observations.query_criteria(
            objectname=target_name,
            obs_collection=mission,
            dataproduct_type="timeseries"
        )
        
        if len(obs_table) == 0:
            return []
        
        # Get data products
        products = Observations.get_product_list(obs_table[:5])  # Limit to first 5
        
        # Filter for light curve files
        lc_products = Observations.filter_products(
            products,
            productSubGroupDescription="LC"
        )
        
        if len(lc_products) == 0:
            return []
        
        # Download
        manifest = Observations.download_products(
            lc_products[:3],
            download_dir=str(self.cache_dir)
        )
        
        results = []
        for file_path in manifest['Local Path']:
            try:
                data = self._read_fits_lightcurve(file_path)
                if data:
                    results.append(data)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
        
        return results
    
    def _read_fits_lightcurve(self, file_path: str) -> Optional[SpectralData]:
        """Read a FITS light curve file."""
        try:
            with fits.open(file_path) as hdul:
                # Try to find the light curve extension
                for ext in hdul:
                    if hasattr(ext, 'columns') and ext.columns is not None:
                        col_names = [c.name.upper() for c in ext.columns]
                        
                        # Look for flux data
                        flux_col = None
                        time_col = None
                        
                        for name in ['PDCSAP_FLUX', 'SAP_FLUX', 'FLUX']:
                            if name in col_names:
                                flux_col = name
                                break
                        
                        for name in ['TIME', 'BTJD', 'BJD']:
                            if name in col_names:
                                time_col = name
                                break
                        
                        if flux_col:
                            flux = ext.data[flux_col]
                            time = ext.data[time_col] if time_col else None
                            
                            # Remove NaN values
                            mask = ~np.isnan(flux)
                            flux = flux[mask]
                            if time is not None:
                                time = time[mask]
                            
                            return SpectralData(
                                wavelength=np.arange(len(flux)),
                                flux=flux,
                                time=time,
                                metadata=dict(hdul[0].header),
                                source=f"FITS/{os.path.basename(file_path)}"
                            )
                
        except Exception as e:
            logger.error(f"Error reading FITS file {file_path}: {e}")
        
        return None
    
    async def fetch_confirmed_exoplanets(
        self, 
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch confirmed exoplanet data from NASA Exoplanet Archive.
        Used for calibration against known discoveries.
        """
        logger.info("Fetching confirmed exoplanets from NASA Exoplanet Archive")
        
        try:
            # Query the Planetary Systems Composite Parameters table
            query = f"""
            SELECT TOP {limit}
                pl_name, hostname, discoverymethod, disc_year,
                pl_orbper, pl_orbsmax, pl_rade, pl_bmasse,
                pl_eqt, st_teff, st_rad, st_mass,
                pl_trandep, pl_trandur, pl_tranmid,
                sy_dist, ra, dec
            FROM pscomppars
            WHERE pl_trandep IS NOT NULL
            ORDER BY disc_year DESC
            """
            
            result = NasaExoplanetArchive.query_criteria(
                table="pscomppars",
                select="*",
                where="pl_trandep is not null",
                order="disc_year desc"
            )
            
            df = result.to_pandas()
            
            # Cache the data
            cache_path = self.cache_dir / "confirmed_exoplanets.csv"
            df.to_csv(cache_path, index=False)
            
            logger.info(f"Fetched {len(df)} confirmed exoplanets")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching exoplanet data: {e}")
            
            # Try to load from cache
            cache_path = self.cache_dir / "confirmed_exoplanets.csv"
            if cache_path.exists():
                logger.info("Loading exoplanet data from cache")
                return pd.read_csv(cache_path)
            
            raise
    
    async def fetch_spectral_data_batch(
        self, 
        targets: List[str],
        mission: str = "kepler"
    ) -> Dict[str, List[SpectralData]]:
        """
        Fetch spectral data for multiple targets in parallel.
        
        Args:
            targets: List of target names
            mission: Mission to query ("kepler", "tess", "hubble")
            
        Returns:
            Dictionary mapping target names to their spectral data
        """
        logger.info(f"Batch fetching {len(targets)} targets from {mission}")
        
        fetch_func = {
            "kepler": self.fetch_kepler_lightcurve,
            "tess": self.fetch_tess_lightcurve,
        }.get(mission.lower())
        
        if not fetch_func:
            raise ValueError(f"Unsupported mission: {mission}")
        
        # Fetch in parallel with rate limiting
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def fetch_with_limit(target: str):
            async with semaphore:
                try:
                    return target, await fetch_func(target)
                except Exception as e:
                    logger.error(f"Error fetching {target}: {e}")
                    return target, []
        
        tasks = [fetch_with_limit(target) for target in targets]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    def load_local_fits(self, file_path: str) -> Optional[SpectralData]:
        """Load spectral data from a local FITS file."""
        return self._read_fits_lightcurve(file_path)
    
    def save_spectral_data(
        self, 
        data: SpectralData, 
        output_path: str,
        format: str = "fits"
    ):
        """
        Save spectral data to file.
        
        Args:
            data: SpectralData object to save
            output_path: Output file path
            format: Output format ("fits", "json", "csv")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "fits":
            # Create FITS file
            primary = fits.PrimaryHDU()
            
            cols = [
                fits.Column(name='WAVELENGTH', format='E', array=data.wavelength),
                fits.Column(name='FLUX', format='E', array=data.flux),
            ]
            
            if data.flux_error is not None:
                cols.append(fits.Column(name='FLUX_ERROR', format='E', array=data.flux_error))
            
            if data.time is not None:
                cols.append(fits.Column(name='TIME', format='E', array=data.time))
            
            table = fits.BinTableHDU.from_columns(cols)
            
            # Add metadata
            for key, value in list(data.metadata.items())[:50]:  # FITS header limit
                if isinstance(value, (str, int, float)):
                    try:
                        table.header[key[:8]] = value
                    except (ValueError, TypeError, KeyError) as e:
                        logger.debug(f"Could not set FITS header '{key}': {e}")
            
            hdul = fits.HDUList([primary, table])
            hdul.writeto(str(output_path), overwrite=True)
            
        elif format == "json":
            with open(output_path, 'w') as f:
                json.dump(data.to_dict(), f, indent=2)
                
        elif format == "csv":
            df = pd.DataFrame({
                'wavelength': data.wavelength,
                'flux': data.flux,
            })
            if data.flux_error is not None:
                df['flux_error'] = data.flux_error
            if data.time is not None:
                df['time'] = data.time
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved spectral data to {output_path}")


class DataPreprocessor:
    """Preprocesses spectral data for model input."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_bins = config.get("bin_count", 1024)
    
    def preprocess(self, data: SpectralData) -> np.ndarray:
        """
        Preprocess spectral data for model input.
        
        Steps:
        1. Remove NaN values
        2. Normalize flux
        3. Remove continuum (optional)
        4. Resample to target bins
        5. Apply sigma clipping
        
        Returns:
            Preprocessed flux array of shape (target_bins,)
        """
        flux = data.flux.copy()
        
        # Remove NaN/Inf
        mask = np.isfinite(flux)
        flux = flux[mask]
        
        if len(flux) == 0:
            return np.zeros(self.target_bins)
        
        # Normalize
        if self.config.get("normalize", True):
            flux = self._normalize(flux)
        
        # Remove continuum
        if self.config.get("remove_continuum", True):
            flux = self._remove_continuum(flux)
        
        # Sigma clipping
        sigma = self.config.get("sigma_clip", 3.0)
        flux = self._sigma_clip(flux, sigma)
        
        # Resample to target bins
        flux = self._resample(flux, self.target_bins)
        
        return flux.astype(np.float32)
    
    def _normalize(self, flux: np.ndarray) -> np.ndarray:
        """Normalize flux to [0, 1] range."""
        f_min, f_max = flux.min(), flux.max()
        if f_max - f_min > 0:
            return (flux - f_min) / (f_max - f_min)
        return flux - f_min
    
    def _remove_continuum(self, flux: np.ndarray) -> np.ndarray:
        """Remove continuum using polynomial fit."""
        x = np.arange(len(flux))

        # Fit a low-order polynomial
        try:
            coeffs = np.polyfit(x, flux, deg=3)
            continuum = np.polyval(coeffs, x)
            return flux - continuum + np.median(flux)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as e:
            logger.debug(f"Continuum removal failed, returning original flux: {e}")
            return flux
    
    def _sigma_clip(self, flux: np.ndarray, sigma: float) -> np.ndarray:
        """Apply sigma clipping to remove outliers."""
        median = np.median(flux)
        std = np.std(flux)
        
        lower = median - sigma * std
        upper = median + sigma * std
        
        flux = np.clip(flux, lower, upper)
        return flux
    
    def _resample(self, flux: np.ndarray, target_size: int) -> np.ndarray:
        """Resample flux to target number of bins."""
        if len(flux) == target_size:
            return flux
        
        # Use interpolation
        x_old = np.linspace(0, 1, len(flux))
        x_new = np.linspace(0, 1, target_size)
        
        return np.interp(x_new, x_old, flux)
    
    def batch_preprocess(self, data_list: List[SpectralData]) -> np.ndarray:
        """Preprocess a batch of spectral data."""
        return np.array([self.preprocess(d) for d in data_list])


# CLI interface
if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description="NASA Data Pipeline")
    parser.add_argument("--source", choices=["mast", "exoplanet_archive"], default="mast")
    parser.add_argument("--target", type=str, help="Target name or list file")
    parser.add_argument("--mission", choices=["kepler", "tess"], default="kepler")
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Initialize pipeline
    pipeline = NASADataPipeline(config.get("nasa", {}), cache_dir=args.output)
    
    # Run async fetch
    async def main():
        if args.source == "exoplanet_archive":
            df = await pipeline.fetch_confirmed_exoplanets()
            print(f"Downloaded {len(df)} confirmed exoplanets")
        else:
            targets = [args.target] if args.target else ["Kepler-186"]
            results = await pipeline.fetch_spectral_data_batch(targets, args.mission)
            for target, data_list in results.items():
                print(f"{target}: {len(data_list)} light curves")
    
    asyncio.run(main())
