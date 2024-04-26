import logging
import signal
import sys
import time

import paho.mqtt.client as mqtt
import pandas as pd
from pvlib import atmosphere, clearsky, irradiance, location

LAT = 43.09176073408273
LON = -73.49606500488254
ALT = 121

logging.basicConfig(
    format="%(asctime)s [%(levelname)s]: %(message)s", level=logging.DEBUG
)


def on_publish(client, userdata, mid, reason_code, properties):
    print("Published", properties)
    try:
        userdata.remove(mid)
    except KeyError:
        print("Race condition detected!")


unacked_publish = set()
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.enable_logger()
client.username_pw_set("brightness", "brightness")

client.on_publish = on_publish

client.user_data_set(unacked_publish)
client.connect("192.168.1.50", 1883, 60)
client.loop_start()


def handler(signum, frame):
    client.disconnect()
    client.loop_stop()
    sys.exit(0)


signal.signal(signal.SIGTERM, handler)

loc = location.Location(latitude=LAT, longitude=LON, altitude=ALT)

while True:

    now = pd.Timestamp.now()

    times = pd.date_range(now, now, tz="America/New_York")

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

    msg_info = client.publish(
        "brightness", irr.T.squeeze().to_json(orient="index"), qos=1
    )
    unacked_publish.add(msg_info.mid)

    while len(unacked_publish):
        time.sleep(0.1)

    msg_info.wait_for_publish()
    time.sleep(1)
