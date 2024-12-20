import torch
from torchvision.transforms import v2
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

normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class ServeModel:
    def __init__(self, model_path, task):
        self.model = UNet(n_channels=3, n_classes=1, bilinear=False)
        state_dict = torch.load(model_path, map_location='cuda:0')
        del state_dict['mask_values']

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(dtype=torch.float16, device='cuda:0')
        
        self.queue = Queue()
        self.queue_condition = threading.Condition()
        self.response_events = {}
        
        self.collection = db[task]
        
        self.thread = threading.Thread(target=self.process_batch, daemon=True)
        self.thread.start()
        
    def analyze(self, img):
        img = Image.open(BytesIO(img)).convert("RGB")
        img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
        assert img.shape == (3, 480, 640), f"Expected shape (3, 480, 640), got {img.shape}"
        if (img > 1).any():
            img = img / 255.0

        input_tensor = torch.tensor(img, dtype=torch.float16)  # Convert to float16

        with torch.inference_mode():
            input_tensor = normalize(input_tensor)
            output = self.model(input_tensor)
            result = (output > 0.5).squeeze(0).cpu().numpy().tolist()
            return result
        
    def process_batch(self):
        while True:
            with self.queue_condition:
                # Wait if the queue is empty
                while self.queue.empty():
                    self.queue_condition.wait()
                
                # Create batch
                batch = []
                request_ids = []
                while not self.queue.empty() and len(batch) < BATCH_SIZE:
                    req = self.queue.get()
                    batch.append(req["input_tensor"])
                    request_ids.append(req["id"])
            
            # Model inference
            with torch.inference_mode():
                # Combine batch into tensor
                batch_tensor = torch.stack(batch).to(dtype=torch.float16, device='cuda:0')
                batch_tensor = normalize(batch_tensor)
                outputs = self.model(batch_tensor)
                
            # Save results and set events
            for idx, request_id in enumerate(request_ids):
                result = (outputs[idx] > 0.5).squeeze(0).cpu().numpy()
                with self.queue_condition:
                    if request_id in self.response_events:
                        self.response_events[request_id]["result"] = result
                        self.response_events[request_id]["event"].set()

    async def check_analysis_result(self, file, request_id):
        if request_id not in self.response_events:
            return
        event = self.response_events[request_id]["event"]
        while True: 
            if event.is_set():
                result = self.response_events.pop(request_id)["result"]
                img_array = np.array(Image.open(BytesIO(file)).convert("RGB"))
                masked_img_array = apply_mask(img_array, result)

                masked_img = Image.fromarray(masked_img_array.astype(np.uint8))
                buffer = BytesIO()
                masked_img.save(buffer, format="PNG")

                buffer.seek(0)
                analyzed_image_id = str(uuid.uuid4())
                s3.upload_fileobj(buffer, "snu-aram", f"analyzed-images/{analyzed_image_id}.png")
                self.collection.find_one_and_update(
                    {"_id": request_id},
                    {"$set": {"analysis_id": f"{analyzed_image_id}.png"}},
                    return_document=True
                )
                return None
            else: 
                await asyncio.sleep(1)

def apply_mask(image_array, mask_array, alpha=0.3):
    mask_array = np.array(mask_array, dtype=np.uint8)
    masks = mask_array != 0

    outlined_image = image_array.copy()

    for c in range(3):
        outlined_image[..., c] = np.where(
            masks,
            (1 - alpha) * image_array[..., c] + alpha * [255, 0, 0][c],
            image_array[..., c]
        )

    return outlined_image

pore_model = ServeModel("pore_model.pth", "pore")
hemo_model = ServeModel("hemo_model.pth", "hemo")

# Batch processing settings
BATCH_SIZE = 8

class Request(BaseModel):
    request_id: str

class Analyzed(BaseModel):
    result: list[list[int]]
    status: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/{skin_type}/images")
def read_images(skin_type: str):
    try:
        collection = db[skin_type]
        images = list(collection.find({}))
        for img in images:
            img["original_image_url"] = f'https://d2o4467n7hkrvd.cloudfront.net/original-images/{img["image_id"]}'
            img["analysis_image_url"] = f'https://d2o4467n7hkrvd.cloudfront.net/analyzed-images/{img["analysis_id"]}'
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/{skin_type}/request", response_model=Request)
async def create_request(skin_type, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        task_model = {"pore": pore_model, "hemo": hemo_model}[skin_type]
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        filename = f"{request_id}{os.path.splitext(file.filename)[1]}"

        file_content = file.file.read()

        # Upload to S3
        s3.upload_fileobj(BytesIO(file_content), "snu-aram", f"original-images/{filename}")

        task_model.collection.insert_one({"_id": request_id, "image_id": filename, "analysis_id": "", "created_at": datetime.now().isoformat()})

        # reset file cursor
        file.file.seek(0)

        # Read and preprocess image
        img = Image.open(file.file).convert("RGB")
        img = np.array(img).transpose(2, 0, 1)  # Convert to (C, H, W)
        assert img.shape == (3, 480, 640), f"Expected shape (3, 480, 640), got {img.shape}"
        if (img > 1).any():
            img = img / 255.0
        
        input_tensor = torch.tensor(img, dtype=torch.float16)  # Convert to float16
        
        # Create event and result storage
        with task_model.queue_condition:
            task_model.response_events[request_id] = {"event": threading.Event(), "result": None}
        
            # Add request to queue
            task_model.queue.put({"id": request_id, "input_tensor": input_tensor})
            task_model.queue_condition.notify()  # Wake up waiting worker thread
        
        background_tasks.add_task(task_model.check_analysis_result, file_content, request_id)
        # Return request ID immediately
        return {"request_id": request_id}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/analyzed/{request_id}")
async def get_analyzed(request_id: str):
    try:
        image = pore_model.find_one({"_id": request_id})
        if image:
            return {**image, "analysis_image_url": f'https://d2o4467n7hkrvd.cloudfront.net/analyzed-images/{image["analysis_id"]}'}
        else:
            image = hemo_model.find_one({"_id": request_id})
            if image:
                return {**image, "analysis_image_url": f'https://d2o4467n7hkrvd.cloudfront.net/analyzed-images/{image["analysis_id"]}'}
            else:
                return {}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

pore_model.thread.join()
hemo_model.thread.join()