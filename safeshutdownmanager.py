import time
from subprocess import check_call

import gpiozero
from notify_run import Notify


class ssdManager:
    def __init__(self):

        self.fail = gpiozero.InputDevice(pin=17)

<<<<<<< Updated upstream
	##########################################
	# runSDManager
	# Starts the manager.
	##########################################
	def runSDManager(self):
		#print "ssdManager::runSDManager:"
		bounceCount = 0
		
		while True:
			time.sleep(2)
			#print "ssdManager::runSDManager: Tick"
			if self.readPowerOffIn() == True:
				bounceCount += 1
			else:
				bounceCount = 0
				
			if bounceCount > 1:
				print("ssdManager::runSDManager: Shutting down now")
				os.system("sudo shutdown -h now")
				#print "DUMMY -- sudo shutdown -h now"
				bounceCount = 0
				time.sleep(20)
				
	
	##########################################
	# readPowerOffIn
	# Returns True if power off input Low indicating power fail.
	##########################################
	def readPowerOffIn(self):
		#print "ssdManager::readPowerOffIn:"
		ret = False
		
		try:
			f = open(SYSFS_POWEROFF_VALUE,'r')
			val = f.read().strip()
			f.close()
			if val == '1':
				ret = True
		except IOError as e:
			#print "ssdManager::readPowerOffIn: Error {} {} {}".format(SYSFS_POWEROFF_VALUE, e.errno, e.strerror)
			pass
=======
        self.run = gpiozero.OutputDevice(pin=19)
        self.run.on()
>>>>>>> Stashed changes

    def runSDManager(self):

        notify = Notify()
        bounceCount = 0

        while True:

            time.sleep(2)

            if self.fail.is_active:
                bounceCount += 1
            else:
                bounceCount = 0

            if bounceCount > 1:
                print("ssdManager::runSDManager: Shutting down now")
                notify.send("Chicken lights shutting down!")
                check_call(["sudo", "poweroff"])
                bounceCount = 0
                time.sleep(20)


if __name__ == "__main__":
    sdm = ssdManager()
    sdm.runSDManager()
