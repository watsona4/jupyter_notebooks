
# coding: utf-8

# In[1]:


import json
from oauth2client import tools
from oauth2client.client import OAuth2WebServerFlow
from oauth2client.file import Storage


# In[2]:


with open('C:\\Users\\watso\\Downloads\\client_id.json') as handle:
    client_json = json.load(handle)
client_id = client_json['installed']['client_id']
client_secret = client_json['installed']['client_secret']


# In[3]:


scope = 'https://www.googleapis.com/auth/spreadsheets'
flow = OAuth2WebServerFlow(client_id, client_secret, scope)


# In[6]:


storage = Storage('credentials.dat')
credentials = storage.get()
if credentials is None or credentials.invalid:
    credentials = tools.run_flow(flow, storage)

