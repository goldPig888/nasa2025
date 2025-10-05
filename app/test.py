#!/usr/bin/env python3
"""CLI helper to download MERRA-2 data for a single city/day."""

import argparse
import sys

import pandas as pd

from lib.merra_fetch import fetch_merra2_day, XR_AVAILABLE


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch one-day MERRA-2 variables for a city.")
    parser.add_argument("city", type=str, help="City name (quoted if spaces)")
    parser.add_argument("date", type=str, help="Date YYYY-MM-DD")
    parser.add_argument("--username", type=str, default=None, help="Earthdata username (defaults to env)")
    parser.add_argument("--password", type=str, default=None, help="Earthdata password (defaults to env)")
    parser.add_argument(
        "--out",
        type=str,
        default="merra2_city_one_day.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not XR_AVAILABLE:
        print("ERROR: xarray/netCDF dependencies missing; install xarray netCDF4 numpy pandas.")
        sys.exit(1)

    result = fetch_merra2_day(
        args.city,
        args.date,
        username=args.username,
        password=args.password,
        verbose=True,
    )

    if not result:
        print("ERROR: Failed to retrieve MERRA-2 data. Check credentials and inputs.")
        sys.exit(1)

    df = pd.DataFrame([result])
    df.to_csv(args.out, index=False)
    print(f"\n💾 Saved CSV: {args.out}")


if __name__ == "__main__":
    main()

