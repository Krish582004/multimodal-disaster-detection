from datetime import datetime, timezone
from celery_app import celery
import openeo
import s3fs
import xarray as xr


@celery.task
def fetch_cdse_sentinel_data(
    location_name: str, bbox: dict, start_date: str, end_date: str
):
    """Fetches Sentinel radar data and saves it with a location-based filename."""
    try:
        connection = openeo.connect("openeo.dataspace.copernicus.eu")
        connection.authenticate_oidc()

        # 1. Load collection
        datacube = connection.load_collection(
            "SENTINEL1_GRD",
            spatial_extent=bbox,
            temporal_extent=[start_date, end_date],
            bands=["VV", "VH"],
        )

        # 2. Calibrate SAR intensity into backscatter (REQUIRED for Sentinel-1 GRD)
        datacube = datacube.sar_backscatter(coefficient="sigma0-ellipsoid")

        # 3. Resample spatial resolution to 20m to prevent memory/timeout errors
        datacube = datacube.resample_spatial(resolution=20)

        # 4. Compute temporal mean
        processed_cube = datacube.mean_time()

        # 5. Save output file
        safe_loc = location_name.replace(" ", "_")
        output_filename = f"{safe_loc}_{start_date}_to_{end_date}.tiff"
        processed_cube.download(output_filename, format="GTiff")

        return {
            "status": "success",
            "source": "CDSE_SENTINEL",
            "file_saved": output_filename,
            "extent_used": bbox,
            "dates_used": [start_date, end_date],
        }

    except Exception as e:
        return {"status": "error", "source": "CDSE_SENTINEL", "message": str(e)}


@celery.task
def fetch_aws_goes_weather():
    """Reads NOAA weather data dynamically using the current UTC time."""
    fs = s3fs.S3FileSystem(anon=True)

    now = datetime.now(timezone.utc)
    year = now.strftime("%Y")
    day_of_year = now.strftime("%j")
    hour = now.strftime("%H")

    bucket_path = f"s3://noaa-goes16/ABI-L2-CMIPF/{year}/{day_of_year}/{hour}/"

    try:
        files = fs.ls(bucket_path)
        if files:
            with fs.open(files[0]) as f:
                ds = xr.open_dataset(f, engine="h5netcdf")
                return {
                    "status": "success",
                    "source": "AWS_GOES",
                    "data_shape": str(ds.dims),
                }
        return {"status": "empty", "message": "No files found for this hour yet."}

    except FileNotFoundError:
        return {
            "status": "pending",
            "message": f"AWS has not generated {bucket_path} yet.",
        }
    except Exception as e:
        return {"status": "error", "source": "AWS_GOES", "message": str(e)}