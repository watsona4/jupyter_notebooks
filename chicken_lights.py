import datetime
import http.client
import json
import logging
import sys
import time
import traceback
import tzlocal

from notify_run import Notify
import suntimes
import yeelight

logging.basicConfig(format='%(asctime)s: %(message)s', level=logging.INFO)

# The following steps are run every morning at 30 mins before sunrise:
#   1. Get sunrise/sunset times for current date
#   2. Compute start time and delay based on current date and time
#   3. Get weather at sunrise
#   4. Set up transitions for fake sunrise (30-min), brightening (1-hr), and sleep (til 1 hr after real sunrise)
#   5. Sleep based on delay time
#   6. Set up and run Flow (auto_on=True), turning off after transition
    
# Get sunrise/sunset times for current date
    
latitude = 43.09176073408273
longitude = -73.49606500488254
    
today = datetime.datetime.today()
logging.info('Today is %s', today)

sun = suntimes.SunTimes(latitude=latitude, longitude=longitude, altitude=114)
logging.info('Sun: %s', sun)
    
sunrise = sun.riselocal(today)
sunset = sun.setlocal(today)

logging.info('Sunrise today: %s', sunrise)
logging.info('Sunset today: %s', sunset)

# Compute start time and delay based on current date and time
    
long_day = sun.durationdelta(datetime.date(2021, 6, 21))
short_day = sun.durationdelta(datetime.date(2021, 12, 21))
    
today_duration = sun.durationdelta(today)
logging.info('Daylight duration today: %s', today_duration)
 
target_day = 2*(today_duration - short_day).total_seconds()/(long_day - short_day).total_seconds() + 14
target_day = datetime.timedelta(hours=target_day)
logging.info('Target duration: %s', target_day)
 
trans_delay = max(1 - (today - datetime.datetime(2021, 12, 8)) / datetime.timedelta(days=28), 0)
logging.info('Transition delay: %s', trans_delay)
  
start_time = sunset - target_day + trans_delay * (target_day - today_duration)
logging.info('Start time: %s', start_time)
 
now = datetime.datetime.now(tz=tzlocal.get_localzone())
logging.info('Time right now: %s', now)

delay = start_time - now - datetime.timedelta(minutes=30)
logging.info('Sleep delay: %s', delay)

if delay.total_seconds() < 0:
    sys.exit()
   
# Set up transitions for fake sunrise, brightening, and sleep until real sunrise

sleep_duration = sunrise - start_time
logging.info('On time following sunrise: %s', sleep_duration)

transitions = [
    yeelight.TemperatureTransition(degrees=2000, duration=50, brightness=0),  # initialize
    yeelight.TemperatureTransition(degrees=3500, duration=30*60*1000, brightness=50),  # sunrise (30 mins)
    yeelight.TemperatureTransition(degrees=5500, duration=60*60*1000, brightness=100),  # brightening (1 hr)
    yeelight.SleepTransition(duration=sleep_duration.total_seconds()*1000),  # sleep until real sunrise
    yeelight.TemperatureTransition(degrees=2000, duration=50, brightness=0),  # finalize
]

flow = yeelight.Flow(count=1, action=yeelight.Flow.actions.off, transitions=transitions)
logging.info('Flow: %s', flow)
logging.info('%s', flow.expression)

# Sleep based on delay time

logging.info('Now sleeping for %d seconds, will continue at %s', delay.total_seconds(), start_time - datetime.timedelta(minutes=30))
time.sleep(delay.total_seconds())

# Set up and run Flow (auto_on=True), turning off after transition

try:
    bulb = yeelight.Bulb('192.168.1.13', auto_on=True)
    logging.info('Bulb found: %s', bulb)
    logging.info('Starting flow %s', flow)
    bulb.start_flow(flow)
except Exception as exc:
    exc_str = traceback.format_exception_only(*sys.exc_info()[:2])
    Notify().send(''.join(exc_str))
    raise
