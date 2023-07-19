import logging
from time import sleep
from datetime import datetime, date, time
from pprint import pprint

import pandas as pd
from notify_run import Notify
from pvlib import location
import requests

import laurel

# Set up logging system
if __debug__:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG
    )
else:
    logging.basicConfig(
        filename="light_control.log",
        format="%(asctime)s: %(message)s",
        level=logging.INFO,
    )

LOC = location.Location(
    latitude=43.09176073408273, longitude=-73.49606500488254, altitude=114
)

# Establish connections to devices

laurel_devices = laurel.laurel("watsona4@gmail.com", "password1")
devices = {}
for device in laurel_devices.devices:
    logging.info("Found device '%s'", device.name)
    devices[device.name] = device

devices["Study Lamp"].network.connect()

# Set up HTTP requests
ARDUINO_ADDR = "http://192.168.1.16"

LUX = []
lights_on = False

THRESHOLD = 40
AZIMUTH = 230

notify = Notify()
notified = False

# Main loop
while True:
    # If off-time, skip
    now = datetime.now()
    logging.debug("Current time: %s", now.time())
    if now.time() >= time(5) and now.time() <= time(22, 30):
        # Get current luminosity and add to list
        try:
            lux = requests.get(ARDUINO_ADDR, timeout=10).json()["lux"]
            logging.debug("Lux value: %f", lux)
            LUX.append(lux)
        except:
            pass
        if len(LUX) >= 4:
            LUX = LUX[1:]
        logging.debug("Lux array: %s", LUX)
            
        # Get average luminosity over last 2 minutes
        if len(LUX) > 0:
            lux = sum(LUX) / len(LUX)
        else:
            lux = 100
        logging.info("Average lux = %f", lux)

        # If below threshold, turn on lights
        if not lights_on and lux < THRESHOLD:
            logging.info("Too dark! (<%f) Turning on lights", THRESHOLD)
            #devices["Garage"].set_power(True)
            #devices["Front Porch"].set_power(True)
            #devices["Kitchen Porch"].set_power(True)
            #devices["Back Porch"].set_power(True)
            #devices["Study Lamp"].set_power(True)
            #devices["Study Lamp"].set_brightness(40)
            #devices["Parlor Lamp"].set_power(True)
            #devices["Parlor Lamp"].set_brightness(40)
            lights_on = True

        if lights_on and lux >= THRESHOLD:
            logging.info("Too bright! (<%f) Turning off lights", THRESHOLD)
            #devices["Garage"].set_power(False)
            #devices["Front Porch"].set_power(False)
            #devices["Kitchen Porch"].set_power(False)
            #devices["Back Porch"].set_power(False)
            #devices["Study Lamp"].set_power(False)
            #devices["Parlor Lamp"].set_power(False)
            lights_on = False

        # Check solar azimuth. If > 220° and it's summer, notify to close windows
        times = pd.date_range(now, now, periods=1, tz="America/New_York")
        solpos = LOC.get_solarposition(times)
        logging.info("Solar azimuth = %.2f, elevation = %.2f", solpos.azimuth, solpos.elevation)
        today = datetime.today().date()
        if (
            not notified
            and solpos.azimuth[0] >= AZIMUTH
            and today >= date(today.year, 4, 15)
            and today <= date(today.year, 10, 15)
        ):
            notify.send("Close west-facing windows")
            notified = True
        if notified and solpos.azimuth[0] < AZIMUTH:
            notified = False

    elif lights_on:
        # Outside on-time and lights are on, turn them off
        logging.info("Bedtime! Turning off lights")
        #devices["Garage"].set_power(False)
        #devices["Front Porch"].set_power(False)
        #devices["Kitchen Porch"].set_power(False)
        #devices["Back Porch"].set_power(False)
        #devices["Study Lamp"].set_power(False)
        #devices["Parlor Lamp"].set_power(False)
        lights_on = False

    # Sleep for 30 seconds
    sleep(30)
