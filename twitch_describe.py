import requests
import json


streamer = 'AdmiralBulldog'  # 30816637
me = 'tracyking96'  # 670275304
'''
response = requests.post('https://id.twitch.tv/oauth2/token'+"?client_id=xw95kqgg65tz3iv1uq5z9gfz8ud2y1&client_secret=ug3q4cxx6yda9k9zca9ve6bzfypeyy&grant_type=client_credentials&scope=user:read:subscriptions")
print(response.json())
'''
# This is one of the route where Twich expose data,
# They have many more: https://dev.twitch.tv/docs'
user_access = 'https://id.twitch.tv/oauth2/authorize'
check_user_sub = "https://api.twitch.tv/helix/subscriptions/user"
check_user = 'https://api.twitch.tv/helix/users'
# In order to authenticate we need to pass our api key through header ,
headers = {"client-id": "xw95kqgg65tz3iv1uq5z9gfz8ud2y1"}#, 'Authorization': 'Bearer '+'hnsq4qglbt87ak0h1e18tpqt4qj08r'}
# Secret = 'ug3q4cxx6yda9k9zca9ve6bzfypeyy'

# The previously set endpoint needs some parameter, here, the Twitcher we want to follow
# Disclaimer, I don't even know who this is, but he was the first one on Twich to have a live stream so I could have nice examples
params = {'client_id':'xw95kqgg65tz3iv1uq5z9gfz8ud2y1', 'redirect_uri':'http://localhost',
          'response_type':'token', 'scope':'user:read:subscriptions', 'state':'hnsq4qglbt87ak0h1e18tpqt4qj08r'}
#params = {'broadcaster_id':'670275304', 'user_id':'30816637'}

#response = requests.get(user_access, headers=headers, params=params)

response = requests.post('https://id.twitch.tv/oauth2/token'+"?client_id=xw95kqgg65tz3iv1uq5z9gfz8ud2y1&client_secret=ug3q4cxx6yda9k9zca9ve6bzfypeyy&code=9q81bw2q68whwila589shmcl64iali&grant_type=authorization_code&redirect_uri=http://localhost") #

# Get token

print(response)
json_response = response.json()
# We get only streams
#streams = json_response.get('data', [])
# We create a small function, (a lambda), that tests if a stream is live or not

print(json_response)

#with open('{}_subscribers.json', 'w') as f:
#    json.dump(json_response, f)
