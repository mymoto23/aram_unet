import json
import requests
from PIL import Image
from io import BytesIO

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbTR1MWliaHowMWR4MDcxZzRxenU0cmdoIiwib3JnYW5pemF0aW9uSWQiOiJjbTR0eTl5dWYwNzFzMDd5MGN0ajQ5OXM2IiwiYXBpS2V5SWQiOiJjbTR1NWc0bTMwYTY4MDczMWc0MDMwemJoIiwic2VjcmV0IjoiOWQwM2MzMzc2MTc1MGU4MjY1MDY4ZjZiYzA4ZGJkNmYiLCJpYXQiOjE3MzQ1NDE4NzUsImV4cCI6MjM2NTY5Mzg3NX0.w7zjoxa4Z5UxjQbpWe9N0BcggQ4tHQlEo85Vp7Em_is"


headers = {
    "Authorization": f"Bearer {API_KEY}",
}

with open("pore.ndjson") as f:
    data_list = [json.loads(line) for line in f]
    
for data in data_list:
    try:
        mask_url = data['projects']['cm4u2fpqh053e070nh54e534k']['labels'][0]['annotations']['objects'][0]['mask']['url']
        response = requests.get(mask_url, headers=headers)
        mask = Image.open(BytesIO(response.content))
        mask.save(f"data/human/pore/masks/{data['data_row']['external_id']}")
    except IndexError:
        dimensions = (data['media_attributes']['width'], data['media_attributes']['height'])
        black = Image.new("L", dimensions, 0)
        black.save(f"data/human/pore/masks/{data['data_row']['external_id']}")