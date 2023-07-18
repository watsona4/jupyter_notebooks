import laurel
import logging
import time

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

# Establish connections to devices


# Set up HTTP requests
# Main loop
while True:

    # If off-time, skip

    # Get current luminosity and add to list

    # Get average luminosity over last 2 minutes

    # If below threshold, turn on lights

    # Check solar azimuth. If > 220° and it's summer, notify to close windows

    # Sleep for 30 seconds

    time.sleep(30)
