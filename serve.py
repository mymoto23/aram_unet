import torch
from PIL import Image
from unet import UNet
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from io import BytesIO
from scipy.ndimage import convolve
from pymongo.mongo_client import MongoClient
from datetime import datetime

import asyncio
import json
import numpy as np
from queue import Queue
import threading
import uuid
import boto3
import os

# S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id="<ACCESS_KEY>",
    aws_secret_access_key="<SECRET_KEY",
    region_name="<REGION>"
)
uri = "<MONGO_DB_URI>"
client = MongoClient(uri, tlsAllowInvalidCertificates=True)
db = client["aram"]
collection = db["original-images"]

# Load model
model = UNet(n_channels=3, n_classes=2, bilinear=False)
state_dict = torch.load('checkpoints/checkpoint_epoch25.pth', map_location='cuda:0')
del state_dict['mask_values']

model.load_state_dict(state_dict)
model.eval()
model = model.to('cuda:0')

# Request queue and condition variable
request_queue = Queue()
queue_condition = threading.Condition()  # Condition variable
response_events = {}  # Store events and results by request ID

# Batch processing settings
BATCH_SIZE = 8

class Request(BaseModel):
    request_id: str

class Analyzed(BaseModel):
    result: list[list[int]]
    status: str        

def process_batch():
    while True:
        with queue_condition:
            # Wait if the queue is empty
            while request_queue.empty():
                queue_condition.wait()
            
            # Create batch
            batch = []
            request_ids = []
            while not request_queue.empty() and len(batch) < BATCH_SIZE:
                req = request_queue.get()
                batch.append(req["input_tensor"])
                request_ids.append(req["id"])
        
        # Combine batch into tensor
        batch_tensor = torch.stack(batch).to('cuda:0')  # Ensure tensor is on GPU
        print(batch_tensor.shape[0])
        # Model inference
        with torch.inference_mode():
            outputs = model(batch_tensor)
        
        # Save results and set events
        for idx, request_id in enumerate(request_ids):
            result = outputs[idx].argmax(dim=0).cpu().numpy().tolist()
            with queue_condition:
                if request_id in response_events:
                    response_events[request_id]["result"] = result
                    response_events[request_id]["event"].set()  # Event complete

def apply_outline_mask(image_array, mask_array):
    mask_array = np.array(mask_array, dtype=np.uint8)

    # Detect the edges of the boxes using a simple edge-detection kernel
    kernel = np.array([[1, 1, 1],
                       [1, -8, 1],
                       [1, 1, 1]])
    edges = convolve(mask_array, kernel, mode="constant", cval=0)
    edges = edges != 0
    outlined_image = image_array.copy()
    outlined_image[edges] = [255, 0, 0]

    return outlined_image


async def check_analysis_result(file, request_id):
    if request_id not in response_events:
        return
    event = response_events[request_id]["event"]
    while True: 
        if event.is_set():
            result = response_events.pop(request_id)["result"]
            img_array = np.array(Image.open(BytesIO(file)).convert("RGB"))
            masked_img_array = apply_outline_mask(img_array, result)

            masked_img = Image.fromarray(masked_img_array.astype(np.uint8))
            buffer = BytesIO()
            masked_img.save(buffer, format="PNG")

            buffer.seek(0)
            analyzed_image_id = str(uuid.uuid4())
            s3.upload_fileobj(buffer, "snu-aram", f"analyzed-images/{analyzed_image_id}.png")
            collection.find_one_and_update(
                {"_id": request_id},
                {"$set": {"analysis_id": f"{analyzed_image_id}.png"}},
                return_document=True
            )
            return None
        else: 
            await asyncio.sleep(1)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/images")
def read_images():
    try:
        images = list(collection.find({}))
        for img in images:
            img["original_image_url"] = f'https://d2o4467n7hkrvd.cloudfront.net/original-images/{img["image_id"]}'
            img["analysis_image_url"] = f'https://d2o4467n7hkrvd.cloudfront.net/analyzed-images/{img["analysis_id"]}'
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/request", response_model=Request)
async def create_request(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        filename = f"{request_id}{os.path.splitext(file.filename)[1]}"

        file_content = file.file.read()

        # Upload to S3
        s3.upload_fileobj(BytesIO(file_content), "snu-aram", f"original-images/{filename}")

        collection.insert_one({"_id": request_id, "image_id": filename, "analysis_id": "", "created_at": datetime.now().isoformat()})

        # reset file cursor
        file.file.seek(0)

        # Read and preprocess image
        img = Image.open(file.file).convert("RGB")
        img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
        assert img.shape == (3, 480, 640), f"Expected shape (3, 480, 640), got {img.shape}"
        if (img > 1).any():
            img = img / 255.0
        
        input_tensor = torch.tensor(img, dtype=torch.float32)  # Convert to float32
        
        # Create event and result storage
        with queue_condition:
            response_events[request_id] = {"event": threading.Event(), "result": None}
        
            # Add request to queue
            request_queue.put({"id": request_id, "input_tensor": input_tensor})
            queue_condition.notify()  # Wake up waiting worker thread
        
        background_tasks.add_task(check_analysis_result, file_content, request_id)
        # Return request ID immediately
        return {"request_id": request_id}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analyzed/{request_id}")
async def get_analyzed(request_id: str):
    try:
        image = collection.find_one({"_id": request_id})
        if image:
            return {**image, "analysis_image_url": f'https://d2o4467n7hkrvd.cloudfront.net/analyzed-images/{image["analysis_id"]}'}
        else:
            return {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Start batch processing thread
threading.Thread(target=process_batch, daemon=True).start()
