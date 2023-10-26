#!/usr/bin/env python
##########################################
#
# safeshutdown.py
# Monitors the power fail/shutdown GPIO input and shuts down.
# Steve Garratt
#
##########################################

import time									# Timer stuff
import os									# System stuff

POWER_OFF_INPUT			= 17
SYSFS_GPIO				= "/sys/class/gpio"
SYSFS_POWEROFF_VALUE	= SYSFS_GPIO + "/gpio" + str(POWER_OFF_INPUT) + "/value"

##########################################
# ssdManager class
# Manages an orderly shutdown.
##########################################

class ssdManager():
	def __init__(self):
		#print "ssdManager::__init__:"
		pass

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
				print "ssdManager::runSDManager: Shutting down now"
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

		#print "ssdManager::readPowerOffIn: [{}] [{}]".format(val, ret)
		return ret
		
		
		
		
##########################################
# startSsdManager
# Starts the manager.
##########################################
def startSsdManager():
	#print "startSsdManager:"
	
	sdm = ssdManager()
	sdm.runSDManager()

if __name__ == '__main__':
	startSsdManager()
