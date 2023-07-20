# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dimond
import requests
from pprint import pprint
import os.path
import pickle

API_TIMEOUT = 5

class LaurelException(Exception):
    pass

def authenticate(username, password):
    """Authenticate with the API and get a token."""
    if not os.path.exists('cync_auth.pkl'):
        API_AUTH = "https://api.gelighting.com/v2/two_factor/email/verifycode"
        auth_data = {'corp_id': "1007d2ad150c4000", 'email': username,
                     'local_lang': 'en-us'}
        r = requests.post(API_AUTH, json=auth_data, timeout=API_TIMEOUT)
        code = input()
        resource = ''.join(random.choice(string.ascii_lowercase) for _ in range(16))
        API_AUTH = "https://api.gelighting.com/v2/user_auth/two_factor"
        auth_data = {'corp_id': "1007d2ad150c4000", 'email': username,
                     'password': password, 'two_factor': code, 'resource': resource}
        r = requests.post(API_AUTH, json=auth_data, timeout=API_TIMEOUT)
        with open('cync_auth.pkl', 'wb') as fileobj:
            pickle.dump(r, fileobj)
    else:
        with open('cync_auth.pkl', 'rb') as fileobj:
            r = pickle.load(fileobj)
    try:
        return (r.json()['access_token'], r.json()['user_id'])
    except KeyError:
        raise(LaurelException('API authentication failed'))


def get_devices(auth_token, user):
    """Get a list of devices for a particular user."""
    API_DEVICES = "https://api.gelighting.com/v2/user/{user}/subscribe/devices"
    headers = {'Access-Token': auth_token}
    r = requests.get(API_DEVICES.format(user=user), headers=headers,
                     timeout=API_TIMEOUT)
    return r.json()

def get_properties(auth_token, product_id, device_id):
    """Get properties for a single device."""
    API_DEVICE_INFO = "https://api.gelighting.com/v2/product/{product_id}/device/{device_id}/property"
    headers = {'Access-Token': auth_token}
    r = requests.get(API_DEVICE_INFO.format(product_id=product_id, device_id=device_id), headers=headers, timeout=API_TIMEOUT)
    return r.json()

    def _get_user(auth_token, user):
        """Get information about the user."""
        API_USER = "https://api.gelighting.com/v2/user/{user}"
        headers = {'Access-Token': auth_token}
        r = requests.get(API_USER.format(user=user), headers=headers, timeout=API_TIMEOUT)
        return r.json()


    (auth_token, user) = _authenticate(username, password)
    user_info = _get_user(auth_token, user)
    devices = _get_devices(auth_token, user)
    for device in devices:
        product_id = device['product_id']
        device_id = device['id']
        device_info = _get_device(auth_token, product_id, device_id)

def callback(link, data):
    if data[7] != 0xdc:
        return
    responses = data[10:18]
    for i in (0, 4):
        response = responses[i:i+4]
        for device in link.devices:
            if device.id == response[0]:
                device.brightness = response[2]
                if device.brightness >= 128:
                  device.brightness = device.brightness - 128
                  device.red = int(((response[3] & 0xe0) >> 5) * 255 / 7)
                  device.green = int(((response[3] & 0x1c) >> 2) * 255 / 7)
                  device.blue = int((response[3] & 0x3) * 255 / 3)
                  device.rgb = True
                else:
                  device.temperature = response[3]
                  device.rgb = False
                if device.callback is not None:
                    device.callback(device.cbargs)

class laurel:
    def __init__(self, user, password):
        (self.auth, self.userid) = authenticate(user, password)
        self.devices = []
        self.networks = []
        mesh_networks = get_devices(self.auth, self.userid)
        for mesh in mesh_networks:
            network = None
            devices = []
            # print('Mesh:')
            # pprint(mesh)
            properties = get_properties(self.auth, mesh['product_id'],
                                        mesh['id'])
            if properties.get('error') is not None:
                continue
            for bulb in properties['bulbsArray']:
                # print('Bulb:')
                # pprint(bulb)
                try:
                    id = int(str(bulb['deviceID'])[-3:])
                    if network is None:
                        network = laurel_mesh(mesh['mac'], mesh['access_key'])
                    device = laurel_device(network, {'name': bulb['displayName'], 'mac': bulb['mac'], 'id': id, 'type': bulb['deviceType'], 'load': 1})
                    network.devices.append(device)
                    self.devices.append(device)
                except KeyError:
                    continue
                
            self.networks.append(network)

class laurel_mesh:
    def __init__(self, address, password):
        self.address = str(address)
        self.password = str(password)
        self.devices = []
        self.link = None

    def connect(self):
        if self.link != None:
            return

        for device in self.devices:
            # Try each device in turn - we only need to connect to one to be
            # on the mesh
            try:                
                self.link = dimond.dimond(0x0211, device.mac, self.address, self.password, self, callback)
                self.link.connect()
                break
            except Exception as e:
                print("Failed to connect to %s" % device.mac, e)
                self.link = None
                pass
        if self.link is None:
            raise(LaurelException("Unable to connect to mesh %s" % self.address))

    def send_packet(self, id, command, params):
        self.link.send_packet(id, command, params)

    def update_status(self):
        self.send_packet(0xffff, 0xda, [])

        
class laurel_device:
    def __init__ (self, network, device):
        self.network = network
        self.name = device['name']
        self.id = device['id']
        self.mac = device['mac']
        self.type = device['type']
        self.load = device['load']
        self.callback = None
        self.brightness = 0
        self.temperature = 0
        self.red = 0
        self.green = 0
        self.blue = 0
        self.rgb = False

    def set_callback(self, callback, cbargs):
        self.callback = callback
        self.cbargs = cbargs

    def set_temperature(self, temperature):
        self.network.send_packet(self.id, 0xe2, [0x05, temperature])
        self.temperature = temperature

    def set_rgb(self, red, green, blue):
        self.network.send_packet(self.id, 0xe2, [0x04, red, green, blue])
        self.red = red
        self.green = green
        self.blue = blue

    def set_brightness(self, brightness):
        self.network.send_packet(self.id, 0xd2, [brightness])
        self.brightness = brightness

    def set_power(self, power):
        self.network.send_packet(self.id, 0xd0, [int(power)])

    def update_status(self):
        self.network.send_packet(self.id, 0xda, [])

    def supports_dimming(self):
        if self.supports_temperature() or \
           self.type == 1 or \
           self.type == 9 or \
           self.type == 17 or \
           self.type == 18 or \
           self.type == 24 or \
           self.type == 81:
            return True

        if self.type == 48 or \
           self.type == 55 or \
           self.type == 56:
            # Switch, depends on load type
            if self.load == 4 or\
               self.load == 5:
                return False
            return True

        return False

    def supports_temperature(self):
        if self.supports_rgb() or \
           self.type == 5 or \
           self.type == 14 or \
           self.type == 15 or \
           self.type == 19 or \
           self.type == 20 or \
           self.type == 28 or \
           self.type == 29 or \
           self.type == 80 or \
           self.type == 83 or \
           self.type == 85 or \
           self.type == 129:
            return True
        return False

    def supports_rgb(self):
        if self.type == 6 or \
           self.type == 7 or \
           self.type == 8 or \
           self.type == 21 or \
           self.type == 22 or \
           self.type == 23 or \
           self.type == 30 or \
           self.type == 31 or \
           self.type == 32 or \
           self.type == 33 or \
           self.type == 34 or \
           self.type == 35 or \
           self.type == 131 or \
           self.type == 132 or \
           self.type == 133:
            return True
        return False
