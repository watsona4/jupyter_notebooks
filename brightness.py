import sys
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import suntimes
from pvlib import atmosphere, location, irradiance, clearsky

from colour_system import CS_HDTV

LAT = 43.09176073408273
LON = -73.49606500488254
ALT = 114


def main():

    today = date.today()

    sun = suntimes.SunTimes(latitude=LAT, longitude=LON, altitude=ALT)

    now = datetime.now()

    times = pd.date_range(now, now, tz="America/New_York")
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
        dni=sky['dni'],
        ghi=sky['ghi'],
        dhi=sky['dhi'],
    )

    print(irr['poa_global'][0])
    

if __name__ == "__main__":
    main()
