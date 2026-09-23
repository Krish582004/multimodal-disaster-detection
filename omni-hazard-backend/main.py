import io
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException  # type: ignore[reportMissingImports]
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from geopy.geocoders import Nominatim
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
import rasterio
from rasterio.plot import show

from celery_app import celery
from database import ScanRecord, SessionLocal
from tasks import fetch_aws_goes_weather, fetch_cdse_sentinel_data

matplotlib.use("Agg")  # Non-GUI backend for server environments

app = FastAPI(title="Omni Hazard Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ScanRequest(BaseModel):
    location_name: str = Field(
        ..., description="Any city or region name (e.g., 'Tokyo', 'London')"
    )
    west: Optional[float] = None
    south: Optional[float] = None
    east: Optional[float] = None
    north: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


def geocode_location(location_name: str) -> dict:
    """Dynamically fetches bounding box for any location name."""
    geolocator = Nominatim(user_agent="omni-hazard-backend")
    location = geolocator.geocode(location_name, geometry="geojson")

    if location and "boundingbox" in location.raw:
        bb = location.raw["boundingbox"]
        return {
            "south": float(bb[0]),
            "north": float(bb[1]),
            "west": float(bb[2]),
            "east": float(bb[3]),
        }
    raise HTTPException(
        status_code=400,
        detail=f"Could not resolve coordinates for '{location_name}'.",
    )


@app.post("/trigger-multi-source-scan")
def trigger_scan(request: ScanRequest):
    # STEP 1: Define start_date and end_date FIRST
    now = datetime.now(timezone.utc).date()
    end_date = request.end_date if request.end_date else now.isoformat()
    start_date = (
        request.start_date
        if request.start_date
        else (now - timedelta(days=5)).isoformat()
    )

    # STEP 2: Define bbox NEXT
    if None in (request.west, request.south, request.east, request.north):
        bbox = geocode_location(request.location_name)
    else:
        bbox = {
            "west": request.west,
            "south": request.south,
            "east": request.east,
            "north": request.north,
        }

    # STEP 3: Now pass the defined variables into Celery tasks
    cdse_task = fetch_cdse_sentinel_data.delay(
        location_name=request.location_name,
        bbox=bbox,
        start_date=start_date,
        end_date=end_date,
    )
    aws_task = fetch_aws_goes_weather.delay()

    # STEP 4: Save record to database
    db = SessionLocal()
    db_record = ScanRecord(
        location_name=request.location_name,
        start_date=start_date,
        end_date=end_date,
        cdse_task_id=cdse_task.id,
        aws_task_id=aws_task.id,
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    db.close()

    return {
        "scan_id": db_record.id,
        "message": f"Scanning '{request.location_name}' from {start_date} to {end_date}",
        "resolved_bbox": bbox,
        "cdse_task_id": cdse_task.id,
        "aws_task_id": aws_task.id,
    }


@app.get("/task-status/{task_id}")
def get_task_status(task_id: str):
    """Checks real-time status of Celery task."""
    task = celery.AsyncResult(task_id)
    response = {"task_id": task_id, "status": task.status}
    if task.ready():
        response["result"] = task.result
    return response


@app.get("/download/{filename}")
def download_file(filename: str):
    """Downloads saved GeoTIFF file."""
    file_path = os.path.join(os.getcwd(), filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail=f"File '{filename}' not found."
        )
    return FileResponse(
        path=file_path, filename=filename, media_type="image/tiff"
    )

@app.get("/view-image/{filename}")
def view_image(filename: str):
    """Reads a saved .tiff file, processes SAR backscatter into decibels (dB),

    and renders a multi-color RGB composite or heat map PNG.
    """
    file_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, detail=f"File '{filename}' not found."
        )

    try:
        with rasterio.open(file_path) as src:
            num_bands = src.count
            fig, ax = plt.subplots(figsize=(10, 10))

            if num_bands >= 2:
                # 1. Read VV (Band 1) and VH (Band 2)
                vv = src.read(1)
                vh = src.read(2)

                # 2. Mask zero/negative values to prevent log errors
                vv = np.where(vv <= 0, np.nan, vv)
                vh = np.where(vh <= 0, np.nan, vh)

                # 3. Convert to Decibels (dB)
                vv_db = 10 * np.log10(vv)
                vh_db = 10 * np.log10(vh)
                ratio_db = vv_db - vh_db  # Polarization ratio

                # 4. Normalize channels to 0-1 range for RGB
                def normalize(band):
                    p2, p98 = np.nanpercentile(band, 2), np.nanpercentile(
                        band, 98
                    )
                    clipped = np.clip(band, p2, p98)
                    return (clipped - p2) / (p98 - p2 + 1e-6)

                r = normalize(vv_db)  # Red channel = VV
                g = normalize(vh_db)  # Green channel = VH
                b = normalize(ratio_db)  # Blue channel = VV/VH Ratio

                # Combine into 3D RGB array
                rgb = np.dstack([
                    np.nan_to_num(r, nan=0.0),
                    np.nan_to_num(g, nan=0.0),
                    np.nan_to_num(b, nan=0.0),
                ])

                ax.imshow(rgb)
                ax.set_title(f"Dual-Pol RGB Radar Composite: {filename}")

            else:
                # Fallback for single-band data using 'viridis' palette
                img_array = src.read(1)
                img_array = np.where(img_array <= 0, np.nan, img_array)
                db_array = 10 * np.log10(img_array)

                vmin = np.nanpercentile(db_array, 2)
                vmax = np.nanpercentile(db_array, 98)

                ax.imshow(db_array, cmap="viridis", vmin=vmin, vmax=vmax)
                ax.set_title(f"Multi-Color Heatmap (dB): {filename}")

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
            buf.seek(0)
            plt.close(fig)

            return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating plot: {str(e)}"
        )

@app.get("/scan-history")
def get_scan_history():
    """Retrieves all past scan records from the SQLite database."""
    db = SessionLocal()
    records = db.query(ScanRecord).order_by(ScanRecord.id.desc()).all()
    db.close()
    return records

