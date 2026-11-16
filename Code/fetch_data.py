"""Fetch and cache Delhi air quality, weather (archive + live IST forecast), and CPCB data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import requests

DELHI_LAT = 28.6139
DELHI_LON = 77.2090
TIMEZONE = "Asia/Kolkata"
IST = timezone(timedelta(hours=5, minutes=30))

AQ_VARIABLES = [
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "carbon_dioxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "aerosol_optical_depth",
    "dust",
    "uv_index",
    "uv_index_clear_sky",
    "methane",
]

WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "boundary_layer_height",
]

DATA_DIR = Path(__file__).resolve().parent / "data"


def get_ist_clock() -> tuple[datetime, str, str, str]:
    """Return current IST datetime, today_ist, tomorrow_ist, and safe archive cutoff."""
    now_ist = datetime.fromtimestamp(time.time(), tz=IST)
    today_ist = now_ist.date()
    tomorrow_ist = today_ist + timedelta(days=1)
    archive_safe_end = today_ist - timedelta(days=7)
    return now_ist, today_ist.isoformat(), tomorrow_ist.isoformat(), archive_safe_end.isoformat()


def fetch_openmeteo_range(
    url: str,
    hourly_vars: list[str],
    chunks: list[tuple[str, str]],
) -> dict:
    """Fetch Open-Meteo hourly data across date chunks and merge chronologically."""
    merged_by_time: dict[str, dict] = {}
    units = {}
    for start_date, end_date in chunks:
        params = {
            "latitude": DELHI_LAT,
            "longitude": DELHI_LON,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": TIMEZONE,
        }
        resp = requests.get(url, params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"Open-Meteo failed for {url} [{start_date}..{end_date}]: "
                f"HTTP {resp.status_code} - {resp.text[:300]}"
            )
        payload = resp.json()
        units = payload.get("hourly_units", units)
        hourly = payload.get("hourly", {})
        times = hourly.get("time", [])
        for idx, ts in enumerate(times):
            merged_by_time[ts] = {var: hourly.get(var, [None] * len(times))[idx] for var in hourly_vars}
        print(f"Fetched {url.split('//')[1].split('/')[0]} [{start_date}..{end_date}]: {len(times)} hours")

    sorted_times = sorted(merged_by_time.keys())
    out_hourly: dict[str, list] = {"time": sorted_times}
    for var in hourly_vars:
        out_hourly[var] = [merged_by_time[ts][var] for ts in sorted_times]
    return {"latitude": DELHI_LAT, "longitude": DELHI_LON, "timezone": TIMEZONE, "hourly_units": units, "hourly": out_hourly}


def fetch_weather_stitched(archive_end: str, tomorrow_ist: str) -> dict:
    """Fetch ERA5 historical weather archive stitched seamlessly with live forecast weather."""
    archive_chunks = [
        ("2022-08-05", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", archive_end),
    ]
    arch = fetch_openmeteo_range("https://archive-api.open-meteo.com/v1/archive", WEATHER_VARIABLES, archive_chunks)

    # Stitch live recent past + today + tomorrow forecast from /v1/forecast
    fc_resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": DELHI_LAT,
            "longitude": DELHI_LON,
            "past_days": 14,
            "forecast_days": 3,
            "hourly": ",".join(WEATHER_VARIABLES),
            "timezone": TIMEZONE,
        },
        timeout=30,
    )
    fc_resp.raise_for_status()
    fc_hourly = fc_resp.json()["hourly"]
    print(f"Fetched api.open-meteo.com/v1/forecast [past 14d + next 3d]: {len(fc_hourly['time'])} hours")

    merged_by_time: dict[str, dict] = {}
    arch_times = arch["hourly"]["time"]
    for idx, ts in enumerate(arch_times):
        merged_by_time[ts] = {var: arch["hourly"][var][idx] for var in WEATHER_VARIABLES}

    max_ts = f"{tomorrow_ist}T23:00"
    fc_times = fc_hourly["time"]
    for idx, ts in enumerate(fc_times):
        if ts <= max_ts:
            # Prefer ERA5 archive where available, fill archive-lag days + tomorrow with forecast
            if ts not in merged_by_time or any(merged_by_time[ts][v] is None for v in WEATHER_VARIABLES):
                merged_by_time[ts] = {var: fc_hourly.get(var, [None] * len(fc_times))[idx] for var in WEATHER_VARIABLES}

    sorted_times = sorted(merged_by_time.keys())
    out_hourly: dict[str, list] = {"time": sorted_times}
    for var in WEATHER_VARIABLES:
        out_hourly[var] = [merged_by_time[ts][var] for ts in sorted_times]
    return {"latitude": DELHI_LAT, "longitude": DELHI_LON, "timezone": TIMEZONE, "hourly": out_hourly}


def fetch_xkdr_cpcb_bonus() -> dict:
    """Load or fetch Oct-Nov 2024 daily Delhi PM2.5 ground monitor readings (CPCB data as archived by xKDR)."""
    out_path = DATA_DIR / "xkdr_cpcb_delhi_oct_nov_2024.csv"
    if out_path.exists() and out_path.stat().st_size > 1000:
        line_count = len(out_path.read_text(encoding="utf-8").strip().splitlines()) - 1
        print(f"Using existing CPCB Delhi ground data: {line_count} rows -> {out_path.name}")
        return {"ok": True, "rows": line_count, "path": str(out_path)}

    url = (
        "https://airquality.xkdr.org/v1/measurements"
        "?city=Delhi&parameter=PM2.5&start=2024-10-01&end=2024-11-30"
        "&agg=daily&format=csv"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.ok and len(resp.text.strip().splitlines()) > 1:
            out_path.write_text(resp.text, encoding="utf-8")
            line_count = len(resp.text.strip().splitlines()) - 1
            print(f"Fetched CPCB Delhi ground data: {line_count} rows -> {out_path.name}")
            return {"ok": True, "rows": line_count, "path": str(out_path)}
        return {"ok": False, "status": resp.status_code}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    now_ist, today_ist, tomorrow_ist, archive_end = get_ist_clock()
    print(f"Current IST time: {now_ist.strftime('%Y-%m-%d %H:%M IST')} | Today IST: {today_ist} | Tomorrow IST: {tomorrow_ist}")

    aq_chunks = [
        ("2022-08-05", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
        ("2026-01-01", tomorrow_ist),
    ]
    aq_data = fetch_openmeteo_range("https://air-quality-api.open-meteo.com/v1/air-quality", AQ_VARIABLES, aq_chunks)
    (DATA_DIR / "openmeteo_aq_hourly.json").write_text(json.dumps(aq_data), encoding="utf-8")

    weather_data = fetch_weather_stitched(archive_end, tomorrow_ist)
    (DATA_DIR / "openmeteo_weather_hourly.json").write_text(json.dumps(weather_data), encoding="utf-8")

    cpcb_status = fetch_xkdr_cpcb_bonus()
    metadata = {
        "fetched_at_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "today_ist": today_ist,
        "tomorrow_ist": tomorrow_ist,
        "aq_hours": len(aq_data["hourly"]["time"]),
        "aq_start": aq_data["hourly"]["time"][0],
        "aq_end": aq_data["hourly"]["time"][-1],
        "weather_hours": len(weather_data["hourly"]["time"]),
        "weather_start": weather_data["hourly"]["time"][0],
        "weather_end": weather_data["hourly"]["time"][-1],
        "cpcb_bonus": cpcb_status,
    }
    (DATA_DIR / "fetch_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Fetch complete:", json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
