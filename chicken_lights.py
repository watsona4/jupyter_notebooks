import argparse
import logging
import signal
import sys
import time

import numpy as np
import paho.mqtt.client as mqtt
import pandas as pd
import suntimes
from pvlib import atmosphere, location, spectrum

from colour_system import CS_HDTV

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.INFO
)

TOPIC = "chicken_lights"

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--mqtt-server", help="IP address of the MQTT server")
PARSER.add_argument("--port", type=int, default=1883, help="MQTT server port to use")
PARSER.add_argument("--username", help="User name to log into MQTT server")
PARSER.add_argument("--password", help="Password to log into MQTT server")
PARSER.add_argument(
    "--timezone", default="America/New_York", help="the timezone to use"
)
PARSER.add_argument(
    "--latitude", type=float, default=43.09176073408273, help="latitude of location"
)
PARSER.add_argument(
    "--longitude", type=float, default=-73.49606500488254, help="longitude of location"
)
PARSER.add_argument(
    "--altitude", type=float, default=121, help="altitude of location in meters"
)

ARGS = PARSER.parse_args()


def on_publish(client, userdata, mid, reason_code, properties):
    logging.debug("Published %s", properties)
    try:
        userdata.remove(mid)
    except KeyError:
        logging.error("Race condition detected!")


CLIENT = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

CLIENT.enable_logger()
CLIENT.username_pw_set(ARGS.username, ARGS.password)

CLIENT.on_publish = on_publish


def handler(signum, frame):
    CLIENT.disconnect()
    CLIENT.loop_stop()
    sys.exit(0)


signal.signal(signal.SIGTERM, handler)


def main():
    unacked_publish = set()

    CLIENT.user_data_set(unacked_publish)
    CLIENT.connect(ARGS.mqtt_server, ARGS.port, 60)
    CLIENT.loop_start()

    today = pd.Timestamp.today(tz=ARGS.timezone)
    logging.info("Today is %s", today)

    sun = suntimes.SunTimes(
        latitude=ARGS.latitude, longitude=ARGS.longitude, altitude=ARGS.altitude
    )
    logging.info("Sun: %s", sun)

    sunrise = pd.Timestamp(sun.riselocal(today)).tz_convert(ARGS.timezone)
    sunset = pd.Timestamp(sun.setlocal(today)).tz_convert(ARGS.timezone)

    logging.info("Sunrise today: %s", sunrise)
    logging.info("Sunset today: %s", sunset)

    dl = today.replace(month=6, day=21)
    ds = today.replace(month=12, day=21)

    dsp = today.replace(month=8, day=15)

    todayp = (dl - dsp) / 2 * (np.cos(np.pi * (dl - today) / (dl - ds)) + 1) + dsp

    start_time = todayp.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    end_time = todayp.replace(hour=23, minute=59, second=59)

    times = pd.date_range(start_time, end_time, freq="10s", tz=ARGS.timezone)

    loc = location.Location(
        latitude=ARGS.latitude, longitude=ARGS.longitude, altitude=ARGS.altitude
    )

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
            for i in range(len(times))
        ]
    )

    norms = np.array([np.linalg.norm(v) for v in spec])
    nanmax = np.nanmax(norms)
    logging.info("Max. irradiance: %s", nanmax)
    brights = norms / nanmax

    spec = np.array([CS_HDTV.spec_to_rgb(s) for s in spec])

    df = pd.DataFrame(
        {
            "Red": spec[:, 0],
            "Green": spec[:, 1],
            "Blue": spec[:, 2],
            "Brightness": brights,
        },
        index=times,
    )

    df.dropna(inplace=True)

    delta_time = df.index[-1] - df.index[0]
    logging.info("Delta time: %s (%s)", delta_time, type(delta_time))

    start_time = sunset - delta_time
    logging.info("Start time: %s", start_time)

    now = pd.Timestamp.now(tz=ARGS.timezone)
    logging.info("Time right now: %s", now)

    delay = start_time - now
    logging.info("Sleep delay: %s", delay)

    if delay.total_seconds() < 0:
        mins = delay.total_seconds() / 6
        logging.info("Stripping first %d entries", int(np.abs(mins)))
        df = df.tail(-int(np.abs(mins)))
    else:
        logging.info(
            "Now sleeping for %d seconds, will continue at %s",
            delay.total_seconds(),
            now + delay,
        )
        time.sleep(delay.total_seconds())

    for idx, row in df.iterrows():
        while pd.Timestamp.now().second % 10 != 0:
            time.sleep(0.5)

        msg1_info = CLIENT.publish(TOPIC + "/status", "on", qos=1)
        unacked_publish.add(msg1_info.mid)

        msg2_info = CLIENT.publish(TOPIC + "/setting", row.to_json(), qos=1)
        unacked_publish.add(msg2_info.mid)

        while len(unacked_publish):
            time.sleep(0.1)

        msg1_info.wait_for_publish()
        msg2_info.wait_for_publish()
        time.sleep(1)

    msg_info = CLIENT.publish(TOPIC + "/status", "off", qos=1)
    unacked_publish.add(msg_info.mid)

    while len(unacked_publish):
        time.sleep(0.1)

    msg_info.wait_for_publish()


if __name__ == "__main__":
    old_day = pd.Timestamp(pd.Timestamp.today().date() - pd.Timedelta(days=1))
    while True:
        logging.info("    old_day: %s", old_day)
        today = pd.Timestamp.today()
        logging.info("    today: %s", today)
        logging.info("        today - old_day = %s", today - old_day)
        if today - old_day >= pd.Timedelta(days=1):
            old_day = today
            main()
        time.sleep(60)
