import sys
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

# Set location as home
LOC = location.Location(
    latitude=43.09176073408273, longitude=-73.49606500488254, altitude=114
)

DEVICES = {}  # Dictionary of devices

ARDUINO_ADDR = "http://192.168.1.16"  # URL of light sensor

THRESHOLD = 40  # Lower lux for turning lights on
AZIMUTH = 230  # Solar apparent azimuth for closing windows

# Gets dictionary of devices and establishes connection to BLE mesh network.
def connect():
    laurel_devices = laurel.laurel("watsona4@gmail.com", "password1")
    DEVICES.clear()
    for device in laurel_devices.devices:
        logging.info("Found device '%s'", device.name)
        DEVICES[device.name] = device

    del DEVICES["Garage"]
    del DEVICES["Front Porch"]

    DEVICES["Study Lamp"].network.connect()

# Turns on all devices
def turn_on():
    for device in DEVICES:
        logging.debug("Turning on %s", device)
        device = DEVICES[device]
        if device.supports_dimming():
            device.set_brightness(40)
        else:
            device.set_power(True)
    
# Turns off all devices
def turn_off():
    for device in DEVICES:
        logging.debug("Turning off %s", device)
        device = DEVICES[device]
        device.set_power(False)

# Main execution function
def main():

    # Notification handle
    notify = Notify()

    # Array of latest lux values
    lux_array = []

    # True if lights have been turned on
    lights_on = False

    # True if close windows notification has been sent
    notified = False

    # Connect to light network
    connect()
    turn_off()
    turn_on()
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
                lux_array.append(lux)
            except:
                pass
            if len(lux_array) >= 4:
                lux_array = lux_array[1:]
            logging.debug("Lux array: %s", lux_array)
            
            # Get average luminosity over last 2 minutes
            if len(lux_array) > 0:
                lux = sum(lux_array) / len(lux_array)
            else:
                lux = THRESHOLD
            logging.info("Average lux = %f", lux)

            # If below threshold, turn on lights
            if not lights_on and lux < THRESHOLD:
                logging.info("Too dark! (<%f) Turning on lights", THRESHOLD)
                turn_on()
                lights_on = True

            if lights_on and lux >= THRESHOLD:
                logging.info("Too bright! (<%f) Turning off lights", THRESHOLD)
                turn_off()
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
            turn_off()
            lights_on = False

        # Sleep for 30 seconds
        sleep(30)

if __name__ == "__main__":
    sys.exit(main())
