import logging
import sys
import time
import traceback
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from notify_run import Notify
import suntimes
import tzlocal
import yeelight
from pvlib import atmosphere, spectrum, location

from colour_system import CS_HDTV

if __debug__:

    CUR_TIME = None

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG
    )

    def set_rgb(self, r, g, b):
        #logging.debug("Setting color to (%3d, %3d, %3d)", r, g, b)
        pass

    yeelight.main.Bulb.set_rgb = set_rgb

    def set_brightness(self, b):
        #logging.debug("Setting brightness to %3d", b)
        pass

    yeelight.main.Bulb.set_brightness = set_brightness

    def turn_off(self):
        logging.debug("Turning off bulb")

    yeelight.main.Bulb.turn_off = turn_off

    def sleep(t):
        global CUR_TIME
        CUR_TIME += timedelta(seconds=t)
        #logging.debug("Sleeping for %g seconds, current time is %s", t, CUR_TIME)

    time.sleep = sleep

else:
    logging.basicConfig(filename="chicken_lights_{}.log".format(date.today().strftime("%Y-%m-%d")),
	format="%(asctime)s: %(message)s", level=logging.INFO)

LAT = 43.09176073408273
LON = -73.49606500488254
ALT = 114

BULB_IP = "192.168.1.13"


def main():

    global CUR_TIME

    # Get sunrise/sunset times for current date

    today = date.today()
    logging.info("Today is %s", today)

    CUR_TIME = datetime.today()

    sun = suntimes.SunTimes(latitude=LAT, longitude=LON, altitude=ALT)
    logging.info("Sun: %s", sun)

    sunrise = sun.riselocal(today)
    sunset = sun.setlocal(today)

    logging.info("Sunrise today: %s", sunrise)
    logging.info("Sunset today: %s", sunset)

    dl = date(today.year, 6, 21)
    ds = date(today.year, 12, 21)

    dsp = date(today.year, 8, 15)

    todayp = (dl - dsp) / 2 * (np.cos(np.pi * (dl - today) / (dl - ds)) + 1) + dsp

    start_time = datetime.fromisoformat(f"{todayp} 00:00:00.000000")
    end_time = datetime.fromisoformat(f"{todayp} 23:59:59.999999")

    times = pd.date_range(start_time, end_time, freq="1min", tz="America/New_York")
    num_times = len(times)

    loc = location.Location(latitude=LAT, longitude=LON, altitude=ALT)

    clearsky = loc.get_clearsky(times)
    solpos = loc.get_solarposition(times)

    relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith)

    spectra = spectrum.spectrl2(
        apparent_zenith=solpos.apparent_zenith,
        aoi=solpos.apparent_zenith,
        surface_tilt=0,
        ground_albedo=0.2,
        surface_pressure=101300,
        relative_airmass=relative_airmass,
        precipitable_water=0.5,
        ozone=0.31,
        aerosol_turbidity_500nm=0.1,
    )

    lam = np.arange(380.0, 781.0, 5)
    spec = np.array(
        [
            np.interp(lam, spectra["wavelength"], spectra["poa_global"][:, i])
            for i in range(num_times)
        ]
    )

    norms = np.array([np.linalg.norm(v) for v in spec])
    nanmax = np.nanmax(norms)
    logging.info("Max. irradiance: %s", nanmax)
    brights = norms / nanmax

    colors = np.array(
        [(t, CS_HDTV.spec_to_rgb(s), b, a, r) for t, s, b, a, r in zip(times, spec, brights, solpos.azimuth, clearsky.ghi)],
        dtype=object,
    )
    colors = colors[np.invert(np.isnan(colors[:, 2].astype(float)))]

    start_time = sunset - (colors[-1][0] - colors[0][0]).to_pytimedelta()
    logging.info("Start time: %s", start_time)

    now = datetime.now(tz=tzlocal.get_localzone())
    logging.info("Time right now: %s", now)

    delay = start_time - now
    logging.info("Sleep delay: %s", delay)

    if delay.total_seconds() < 0:
        mins = delay.total_seconds() / 60
        logging.info("Stripping first %d entries", int(np.abs(mins)))
        colors = colors[int(np.abs(mins)) :]
    else:
        logging.info(
            "Now sleeping for %d seconds, will continue at %s",
            delay.total_seconds(),
            now + delay,
        )
        time.sleep(delay.total_seconds())

    # Set up and run flow, turning off after transition

    bulb = yeelight.Bulb(BULB_IP, auto_on=True)
    logging.info("Bulb found: %s", bulb)

    notify = Notify()
    notified = False

    for _, color, bright, azimuth, irradiance in colors:

        color = list(map(int, (255 * color).round()))

        if not notified and azimuth > 200 and irradiance > 880:
            notify.send('Close west-facing windows')
            notified = True

        bright = int(round(100 * bright))

        try:
            logging.info("color = %s, brightness = %d, azimuth = %.2f, irradiance = %.2f", color, bright, azimuth, irradiance)
            bulb.set_rgb(*color)
            bulb.set_brightness(bright)
        except:
            traceback.print_exc()

        time.sleep(60)

    logging.info("Turning bulb off")
    bulb.turn_off()


if __name__ == "__main__":
    main()
