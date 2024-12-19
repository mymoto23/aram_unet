import requests
from PIL import Image
import numpy as np
import logging
from time import sleep

url = "http://127.0.0.1:7777"
image_path = "data/imgs/20230305112206155.jpg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(f"{url}/request", files=files)

request_id = response.json().get("request_id")

response = requests.get(f"{url}/analyzed/{request_id}")
print(response.json())
sleep(5)
response = requests.get(f"{url}/analyzed/{request_id}")
result = np.array(response.json().get("result"))
print(result.shape)
result = Image.fromarray(result.astype(np.uint8) * 255)

# save the image
result.save("output.jpg")
