import sys
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import suntimes
from pvlib import atmosphere, location, irradiance, clearsky

LAT = 43.09176073408273
LON = -73.49606500488254
ALT = 114


def main():
    today = date.today()

    sun = suntimes.SunTimes(latitude=LAT, longitude=LON, altitude=ALT)

    start_time = datetime.fromisoformat("2024-01-01 00:00:00.000")
    end_time = datetime.fromisoformat("2024-12-31 23:59:59.999")

    times = pd.date_range(start_time, end_time, freq="1min", tz="America/New_York")
    num_times = len(times)

    loc = location.Location(latitude=LAT, longitude=LON, altitude=ALT)

    solpos = loc.get_solarposition(times)

    relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith)
    absolute_airmass = atmosphere.get_absolute_airmass(relative_airmass)

    linke_turbidity = clearsky.lookup_linke_turbidity(times, LAT, LON)

    sky = clearsky.ineichen(
        apparent_zenith=solpos.apparent_zenith,
        airmass_absolute=absolute_airmass,
        linke_turbidity=linke_turbidity,
        altitude=ALT,
    )

    irr = irradiance.get_total_irradiance(
        surface_tilt=90,
        surface_azimuth=180,
        solar_zenith=solpos.zenith,
        solar_azimuth=solpos.azimuth,
        dni=sky["dni"],
        ghi=sky["ghi"],
        dhi=sky["dhi"],
    )

    #irr.to_hdf("brightness.h5", key="data", mode="w", format="table", data_columns=True)
    irr["poa_global"].to_pickle("brightness.pkl")

if __name__ == "__main__":
    main()
