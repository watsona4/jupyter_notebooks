from datetime import date, datetime, timedelta
import logging
import sys
import time

import numpy as np
import suntimes
import tzlocal
import yeelight
from scipy.constants import c, h, k

from colour_system import CS_HDTV

if __debug__:

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG
    )

    def set_rgb(self, r, g, b):
        logging.debug("Setting color to (%3d, %3d, %3d)", r, g, b)

    yeelight.main.Bulb.set_rgb = set_rgb

    def set_brightness(self, b):
        logging.debug("Setting brightness to %3d", b)

    yeelight.main.Bulb.set_brightness = set_brightness

    def turn_off(self):
        logging.debug("Turning off bulb")

    yeelight.main.Bulb.turn_off = turn_off

    def sleep(t):
        logging.debug("Sleeping for %g seconds", t)

    time.sleep = sleep

else:
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO)

LAT = 43.09176073408273
LON = -73.49606500488254
ALT = 114

BULB_IP = "192.168.1.13"


def planck(lam, T):
    """ Returns the spectral radiance of a black body at temperature T.

    Returns the spectral radiance, B(lam, T), in W.sr-1.m-2 of a black body
    at temperature T (in K) at a wavelength lam (in nm), using Planck's law.

    """

    lam_m = lam / 1.0e9
    fac = h * c / lam_m / k / T
    B = 2 * h * c ** 2 / lam_m ** 5 / (np.exp(fac) - 1)
    return B


def convert_K_to_RGB(colour_temperature):

    lam = np.arange(380.0, 781.0, 5)
    spec = planck(lam, colour_temperature)
    rgb = CS_HDTV.spec_to_rgb(spec)
    rgb = (255 * rgb).astype(int)

    return rgb


def set_values(bulb, rgb, bright):
    bulb.set_rgb(*map(int, rgb))
    bulb.set_brightness(int(bright))


def run_flow(bulb, temp_range, bright_range, num_steps, sleep_duration=60):
    for temp, bright in zip(np.linspace(*temp_range, num_steps),
                            np.linspace(*bright_range, num_steps)):
        set_values(bulb, convert_K_to_RGB(temp), bright)
        time.sleep(sleep_duration)


def main():

    # The following steps are run every morning at 30 mins before sunrise:
    #   1. Get sunrise/sunset times for current date
    #   2. Compute start time and delay based on current date and time
    #   3. Get weather at sunrise
    #   4. Set up transitions for fake sunrise (30-min), brightening (1-hr), and sleep (til 1 hr after real sunrise)
    #   5. Sleep based on delay time
    #   6. Set up and run Flow (auto_on=True), turning off after transition

    # Get sunrise/sunset times for current date

    today = datetime.today()
    logging.info("Today is %s", today)

    sun = suntimes.SunTimes(latitude=LAT, longitude=LON, altitude=ALT)
    logging.info("Sun: %s", sun)

    sunrise = sun.riselocal(today)
    sunset = sun.setlocal(today)
    midday = (sunrise + sunset) / 2

    logging.info("Sunrise today: %s", sunrise)
    logging.info("Sunset today: %s", sunset)
    logging.info('Midday today: %s', midday)

    # Compute start time and delay based on current date and time

    long_day = sun.durationdelta(date(2021, 6, 21))
    short_day = sun.durationdelta(date(2021, 12, 21))

    today_duration = sun.durationdelta(today)
    logging.info("Daylight duration today: %s", today_duration)

    today_excess = (today_duration - short_day).total_seconds()
    max_excess = (long_day - short_day).total_seconds()
    target_day_hours = 2 * today_excess / max_excess + 14
    target_day = timedelta(hours=target_day_hours)
    logging.info("Target duration: %s", target_day)

    trans_delay = max(1 - (today - datetime(2021, 12, 8)) / timedelta(days=28), 0,)
    logging.info("Transition delay: %s", trans_delay)

    start_time = sunset - target_day + trans_delay * (target_day - today_duration)
    logging.info("Start time: %s", start_time)

    morning = midday - start_time - datetime.timedelta(minutes=60)
    afternoon = sunset - midday - datetime.timedelta(minutes=60)
    logging.info('Morning duration: %s', morning)
    logging.info('Afternoon duration: %s', afternoon)

    now = datetime.now(tz=tzlocal.get_localzone())
    logging.info("Time right now: %s", now)

    delay = start_time - now - timedelta(minutes=30)
    logging.info("Sleep delay: %s", delay)

    if not __debug__ and delay.total_seconds() < 0:
        logging.info("Run in the past. Exiting.")
        sys.exit()

    # Sleep based on delay time

    logging.info(
        "Now sleeping for %d seconds, will continue at %s",
        delay.total_seconds(),
        start_time - timedelta(minutes=30),
    )
    time.sleep(delay.total_seconds())

    # Set up and run flow, turning off after transition

    bulb = yeelight.Bulb(BULB_IP, auto_on=True)
    logging.info("Bulb found: %s", bulb)

    # 30 minutes from 2000K to 3500K and from 0 brightness to 50 brightness
    run_flow(bulb, (2000, 3500), (0, 50), 30)

    # 60 minutes from 3500K to 5500K and from 50 brightness to 100 brightness
    run_flow(bulb, (3500, 5500), (50, 100), 60)

    sleep_duration = sunrise - start_time + timedelta(minutes=90)
    logging.info("On time following sunrise: %s", sleep_duration)

    logging.info(
        "Now sleeping for %d seconds, will continue at %s",
        sleep_duration.total_seconds(),
        datetime.now() + sleep_duration,
    )

    time.sleep(sleep_duration.total_seconds())

    logging.info("Resetting bulb")
    set_values(bulb, convert_K_to_RGB(2000), 0)

    logging.info("Turning bulb off")
    bulb.turn_off()


if __name__ == "__main__":
    main()
