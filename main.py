from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import faiss
import numpy as np
import boto3
from PIL import Image
import io
import zipfile
import os
import json
from typing import List
import uuid
from datetime import datetime
from dotenv import load_dotenv
import cv2
from enum import Enum

# InsightFace (ArcFace + RetinaFace)
from insightface.app import FaceAnalysis

# Import CSV handler
from csv_handler import CSVHandler

import hashlib
# from functools import lru_cache

search_cache = {}

load_dotenv()

app = FastAPI()
VIDEO_CACHE_PATH = "video_cache.json"
# ===================== CORS =====================

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ===================== CONFIG =====================

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME")

FAISS_INDEX_PATH = "face_embeddings.index"
METADATA_PATH = "face_metadata.json"
JOBS_PATH = "jobs.json"

# ArcFace (InsightFace) embedding dimension
EMBEDDING_DIM = 512
S3_IMAGES_PREFIX = "images/"

# ===================== S3 =====================

s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

# ===================== CSV Handler =====================

csv_handler = CSVHandler(s3_client, S3_BUCKET, AWS_REGION)

# ===================== InsightFace (CPU) =====================

"""
We use InsightFace FaceAnalysis with ArcFace-based recognition.
- providers=['CPUExecutionProvider'] guarantees CPU-only ONNX Runtime.
- det_size controls detection resolution.
"""

face_app = FaceAnalysis(
    name="buffalo_l",           # good general-purpose model set
    providers=['CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ===================== FAISS Index & Metadata =====================

"""
We use cosine similarity via Inner Product (IP) FAISS index:
- InsightFace embeddings are already L2-normalized (normed_embedding)
- So inner product == cosine similarity in [-1, 1]
"""

if os.path.exists(FAISS_INDEX_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata = []

# ===================== Job tracking =====================

if os.path.exists(JOBS_PATH):
    with open(JOBS_PATH, 'r') as f:
        jobs = json.load(f)
else:
    jobs = {}


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ===================== Helpers: Persistence =====================

def save_index():
    """Persist FAISS index + metadata."""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f)


def save_jobs():
    """Persist jobs"""
    with open(JOBS_PATH, 'w') as f:
        json.dump(jobs, f)


def update_job_status(job_id: str, status: JobStatus, **kwargs):
    """Update and persist a job status."""
    jobs[job_id].update({
        'status': status,
        'updated_at': datetime.now().isoformat(),
        **kwargs
    })
    save_jobs()


# ===================== Helpers: Face Embeddings (InsightFace) =====================

def extract_face_embeddings_from_rgb(rgb_image: np.ndarray) -> List[np.ndarray]:
    """
    Extract face embeddings from a RGB image array using InsightFace.
    Returns list of 512-dim normalized embeddings.
    """
    # InsightFace expects BGR uint8
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr_image)

    embeddings = []
    for face in faces:
        # face.normed_embedding is already L2-normalized
        emb = face.normed_embedding.astype(np.float32)
        embeddings.append(emb)

    return embeddings


def extract_face_embeddings(image_bytes: bytes) -> List[np.ndarray]:
    """
    Extract embeddings directly from raw image bytes.
    Used for bulk ZIP ingestion.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        rgb = np.array(image)
        embeddings = extract_face_embeddings_from_rgb(rgb)
        return embeddings
    except Exception:
        return []


# ===================== Helpers: S3 Upload =====================

def upload_to_s3(image_bytes: bytes, original_filename: str) -> str:
    """Upload image bytes to S3 and return the public URL."""
    try:
        file_ext = os.path.splitext(original_filename)[1] or '.jpg'
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        s3_key = f"{S3_IMAGES_PREFIX}{unique_filename}"

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=image_bytes,
            ContentType='image/jpeg'
        )

        url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        return url

    except Exception as e:
        raise Exception(f"S3 upload failed: {e}")


# ===================== Background ZIP Processing =====================
async def process_zip_in_background(job_id: str, zip_path: str):
    try:
        update_job_status(job_id, JobStatus.PROCESSING, progress=0)

        processed_images = []
        failed_images = []
        total_faces = 0

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            all_files = [
                f for f in zip_ref.namelist()
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            total_files = len(all_files)

            for idx, filename in enumerate(all_files):
                try:
                    # Read image bytes from ZIP file
                    with zip_ref.open(filename) as img_file:
                        image_bytes = img_file.read()

                    # Extract embeddings
                    embeddings = extract_face_embeddings(image_bytes)

                    if not embeddings:
                        failed_images.append({
                            "filename": filename,
                            "reason": "No faces detected"
                        })
                        continue

                    # Upload image to S3
                    s3_url = upload_to_s3(image_bytes, filename)

                    # Store each detected face embedding
                    for face_idx, emb in enumerate(embeddings):
                        emb = emb.reshape(1, -1)
                        index.add(emb)

                        metadata.append({
                            "image_url": s3_url,
                            "original_filename": os.path.basename(filename),
                            "uploaded_at": datetime.now().isoformat(),
                            "face_index": face_idx,
                            "face_count": len(embeddings),
                            "job_id": job_id
                        })

                        total_faces += 1

                    processed_images.append({
                        "filename": filename,
                        "faces_detected": len(embeddings),
                        "s3_url": s3_url
                    })

                except Exception as e:
                    failed_images.append({
                        "filename": filename,
                        "reason": str(e)
                    })

                # update progress
                progress = int(((idx + 1) / total_files) * 100)
                update_job_status(job_id, JobStatus.PROCESSING, progress=progress)

        # Save index & metadata
        if total_faces > 0:
            save_index()

        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=100,
            total_images_processed=len(processed_images),
            total_images_failed=len(failed_images),
            total_faces=total_faces,
            processed_images=processed_images,
            failed_images=failed_images
        )

    except Exception as e:
        update_job_status(job_id, JobStatus.FAILED, error=str(e))

    finally:
        # Always delete temp ZIP file
        if os.path.exists(zip_path):
            os.remove(zip_path)



def load_video_cache():
    if os.path.exists(VIDEO_CACHE_PATH) and os.path.getsize(VIDEO_CACHE_PATH) > 2:
        try:
            with open(VIDEO_CACHE_PATH, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_video_cache(cache: dict):
    with open(VIDEO_CACHE_PATH, "w") as f:
        json.dump(cache, f)

video_cache = load_video_cache()




# ===================== FACE RECOGNITION ENDPOINTS =====================
@app.post("/upload-faces")
async def upload_faces(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a ZIP file")

    try:
        job_id = str(uuid.uuid4())
        zip_path = f"/tmp/{job_id}.zip"

        # SAVE ZIP IN CHUNKS (NOT IN RAM)
        with open(zip_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024 * 2)  # 2MB chunks
                if not chunk:
                    break
                f.write(chunk)

        # Register job
        jobs[job_id] = {
            "job_id": job_id,
            "status": JobStatus.PENDING,
            "filename": file.filename,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "progress": 0,
            "zip_path": zip_path
        }
        save_jobs()

        background_tasks.add_task(process_zip_in_background, job_id, zip_path)

        return {
            "status": "received",
            "job_id": job_id,
            "message": "Upload complete. Processing started."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")



@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Check background job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse(jobs[job_id])


@app.get("/jobs")
async def list_jobs():
    """List all jobs."""
    return JSONResponse({
        'total_jobs': len(jobs),
        'jobs': list(jobs.values())
    })


@app.post("/search-face-no-cache")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = 20,
    similarity_threshold: float = 0.4  # Typical ArcFace cosine threshold
):
    """
    Search for similar faces.
    - Uses InsightFace (ArcFace) embeddings.
    - FAISS cosine similarity (inner product).
    - similarity_threshold: 0.3–0.5 is typical.
    """
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No faces in database")

    try:
        image_bytes = await file.read()

        # Basic validation + quality checks
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        width, height = image.size
        if width < 100 or height < 100:
            raise HTTPException(
                status_code=400,
                detail="Image resolution too low. Please upload minimum 100x100 pixels."
            )

        rgb = np.array(image)

        # Sharpness check (variance of Laplacian)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Image sharpness: {laplacian_var}")

        if laplacian_var < 50:
            raise HTTPException(
                status_code=400,
                detail="Image is too blurry. Please upload a clearer, sharper image."
            )

        # Extract face embeddings from the query image
        embeddings = extract_face_embeddings_from_rgb(rgb)

        if not embeddings:
            if laplacian_var < 100:
                raise HTTPException(
                    status_code=400,
                    detail="No clear face detected. Image appears blurry. Upload sharper image."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No face detected. Ensure: face is visible, good lighting, not too far."
                )

        print(f"Query: {len(embeddings)} faces detected")

        all_results = []

        for query_idx, query_embedding in enumerate(embeddings):
            query_vec = query_embedding.reshape(1, -1)
            # Request more than top_k to allow deduplication later
            k = min(top_k * 2, index.ntotal)

            similarities, indices = index.search(query_vec, k)

            seen_urls = set()

            for sim, idx_val in zip(similarities[0], indices[0]):
                if idx_val < 0 or idx_val >= len(metadata):
                    continue

                # sim is cosine similarity in [-1, 1]
                similarity = float(sim)

                if similarity >= similarity_threshold:
                    item = metadata[idx_val]
                    image_url = item['image_url']

                    if image_url not in seen_urls:
                        seen_urls.add(image_url)
                        all_results.append({
                            'image_url': image_url,
                            'original_filename': item['original_filename'],
                            'similarity_score': similarity,
                            # For convenience, define a "distance"-like value
                            'distance': float(1.0 - similarity),
                            'uploaded_at': item['uploaded_at']
                        })

        # Sort by similarity descending
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)

        # Deduplicate by URL and cut to top_k
        unique_results = []
        seen_urls = set()
        for result in all_results:
            if result['image_url'] not in seen_urls and len(unique_results) < top_k:
                seen_urls.add(result['image_url'])
                unique_results.append(result)

        # Quality label
        if laplacian_var > 200:
            quality_status = "excellent"
        elif laplacian_var > 100:
            quality_status = "good"
        else:
            quality_status = "acceptable"

        return JSONResponse({
            'status': 'success',
            'query_faces_detected': len(embeddings),
            'image_quality': {
                'sharpness_score': float(laplacian_var),
                'quality_status': quality_status,
                'resolution': f"{width}x{height}"
            },
            'total_matches': len(unique_results),
            'similarity_threshold': similarity_threshold,
            'results': unique_results
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

def hash_embedding(emb: np.ndarray) -> str:
    """Hash embedding to detect repeated queries."""
    return hashlib.md5(emb.tobytes()).hexdigest()


def resize_for_detection(rgb):
    """Downscale large images for faster face detection."""
    h, w = rgb.shape[:2]
    scale = min(640 / h, 640 / w, 1.0)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
    return rgb

# @app.post("/search-face")
# async def search_face(
#     file: UploadFile = File(...),
#     top_k: int = 20,
#     similarity_threshold: float = 0.4
# ):
#     """Optimized search-face endpoint with caching + faster FAISS search."""
#     if index.ntotal == 0:
#         raise HTTPException(status_code=400, detail="No faces in database")

#     try:
#         image_bytes = await file.read()

#         # Load + validate image
#         try:
#             image = Image.open(io.BytesIO(image_bytes))
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid image file")

#         if image.mode != 'RGB':
#             image = image.convert('RGB')

#         rgb = np.array(image)
#         rgb = resize_for_detection(rgb)  # 🔥 Speed boost for detection

#         # Extract embeddings
#         embeddings = extract_face_embeddings_from_rgb(rgb)
#         if not embeddings:
#             raise HTTPException(status_code=400, detail="No face detected")

#         # Use only first face for search (as before)
#         query_emb = embeddings[0]

#         # ------------ 🔥 CACHING LOGIC ------------
#         emb_hash = hash_embedding(query_emb)

#         if emb_hash in search_cache:
#             cached = search_cache[emb_hash]

#             # If user asks for equal or smaller top_k → return cached
#             if cached["top_k"] >= top_k:
#                 return cached["result"]

#         # ------------ 🔥 Batch FAISS Search ------------
#         # Even though we have 1 face, batching speeds up FAISS
#         query_batch = np.stack([query_emb]).astype('float32')

#         # Reduce FAISS workload
#         k = min(top_k, index.ntotal)

#         similarities, indices = index.search(query_batch, k)

#         results = []
#         seen_urls = set()

#         # Single row (query_batch size = 1)
#         for sim, idx_val in zip(similarities[0], indices[0]):
#             if idx_val < 0 or idx_val >= len(metadata):
#                 continue
#             if sim < similarity_threshold:
#                 continue

#             item = metadata[idx_val]
#             url = item["image_url"]
#             if url in seen_urls:
#                 continue

#             seen_urls.add(url)
#             results.append({
#                 "image_url": url,
#                 "original_filename": item["original_filename"],
#                 "similarity_score": float(sim),
#                 "distance": float(1.0 - sim),
#                 "uploaded_at": item["uploaded_at"]
#             })

#         # ------------ 🔥 Sort only once (fast) ------------
#         results.sort(key=lambda x: x["similarity_score"], reverse=True)
#         results = results[:top_k]

#         response_data = {
#             "status": "success",
#             "query_faces_detected": len(embeddings),
#             "total_matches": len(results),
#             "similarity_threshold": similarity_threshold,
#             "results": results
#         }

#         # ------------ 🔥 Save to cache ------------
#         search_cache[emb_hash] = {
#             "top_k": top_k,
#             "result": response_data
#         }

#         return response_data

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/search-face")
async def search_face(
    file: UploadFile = File(...),
    top_k: int = 20,
    similarity_threshold: float = 0.4
):
    """Optimized search-face endpoint with caching + faster FAISS search."""
    if index.ntotal == 0:
        raise HTTPException(status_code=400, detail="No faces in database")

    try:
        image_bytes = await file.read()

        # Load + validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        rgb = np.array(image)
        rgb = resize_for_detection(rgb)

        # Extract embeddings
        embeddings = extract_face_embeddings_from_rgb(rgb)
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected")

        query_emb = embeddings[0]
        emb_hash = hash_embedding(query_emb)

        if emb_hash in search_cache:
            cached = search_cache[emb_hash]
            if cached["top_k"] >= top_k:
                return cached["result"]

        query_batch = np.stack([query_emb]).astype('float32')
        
        # 🔥 Request more results to account for deleted items
        k = min(top_k * 3, index.ntotal)  # Get 3x more to filter deleted
        
        similarities, indices = index.search(query_batch, k)

        results = []
        seen_urls = set()

        for sim, idx_val in zip(similarities[0], indices[0]):
            if idx_val < 0 or idx_val >= len(metadata):
                continue
            if sim < similarity_threshold:
                continue

            item = metadata[idx_val]
            
            # 🔥 CRITICAL: Skip deleted items
            if item.get('deleted', False):
                continue
            
            url = item["image_url"]
            if url in seen_urls:
                continue

            seen_urls.add(url)
            results.append({
                "image_url": url,
                "original_filename": item["original_filename"],
                "similarity_score": float(sim),
                "distance": float(1.0 - sim),
                "uploaded_at": item["uploaded_at"]
            })
            
            # 🔥 Stop once we have enough results
            if len(results) >= top_k:
                break

        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:top_k]

        response_data = {
            "status": "success",
            "query_faces_detected": len(embeddings),
            "total_matches": len(results),
            "similarity_threshold": similarity_threshold,
            "results": results
        }

        search_cache[emb_hash] = {
            "top_k": top_k,
            "result": response_data
        }

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics."""
    unique_images = len(set(m['image_url'] for m in metadata)) if metadata else 0
    return {
        'total_embeddings': index.ntotal,
        'total_unique_images': unique_images,
        'index_dimension': EMBEDDING_DIM,
        'storage_path': f"s3://{S3_BUCKET}/{S3_IMAGES_PREFIX}"
    }


@app.get("/list-images")
async def list_images():
    """List all stored images (unique by URL)."""
    unique_images = {}

    for item in metadata:
        url = item['image_url']
        if url not in unique_images:
            unique_images[url] = {
                'image_url': url,
                'original_filename': item['original_filename'],
                'uploaded_at': item['uploaded_at'],
                'total_faces': item['face_count']
            }

    return JSONResponse({
        'status': 'success',
        'total_images': len(unique_images),
        'storage_path': f"s3://{S3_BUCKET}/{S3_IMAGES_PREFIX}",
        'images': list(unique_images.values())
    })


@app.delete("/reset")
async def reset_database():
    """
    Reset FAISS index and metadata.
    Note: This will clear all face embeddings (but not S3 images).
    """
    global index, metadata
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    metadata = []
    save_index()
    return {'status': 'Database reset successfully'}


# ===================== CSV ENDPOINTS =====================

@app.post("/update-csv-record")
async def update_csv_record(
    file: UploadFile = File(...),
    ada_no: str = None,
    new_number: str = None,
    csv_file: str = None
):
    """
    Update CSV record with new phone number and photo.
    - csv_file must be 'thailand_data_bangkok' or 'thailand_data_phuket'
    """
    if not ada_no:
        raise HTTPException(status_code=400, detail="ada_no is required")

    if not new_number:
        raise HTTPException(status_code=400, detail="new_number is required")

    if csv_file not in ["thailand_data_bangkok", "thailand_data_phuket"]:
        raise HTTPException(
            status_code=400,
            detail="csv_file must be 'thailand_data_bangkok' or 'thailand_data_phuket'"
        )

    try:
        s3_url = await csv_handler.upload_image_to_s3(file, ada_no)
        updated_record = csv_handler.update_record(csv_file, ada_no, new_number, s3_url)

        if not updated_record:
            return JSONResponse({
                'status': 'skipped',
                'message': (
                    f"ada_no '{ada_no}' not found in {csv_file}.csv. No changes made."
                ),
                'ada_no': ada_no,
                'csv_file': csv_file
            })

        return JSONResponse({
            'status': 'success',
            'message': f"Successfully updated record for ada_no '{ada_no}'",
            'ada_no': ada_no,
            'csv_file': csv_file,
            's3_url': s3_url,
            'updated_record': updated_record
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.get("/view-csv-record/{csv_file}/{ada_no}")
async def view_csv_record(csv_file: str, ada_no: str):
    """View specific CSV record."""
    try:
        record = csv_handler.get_record(csv_file, ada_no)

        if not record:
            raise HTTPException(status_code=404, detail=f"ada_no '{ada_no}' not found")

        return JSONResponse({
            'status': 'success',
            'csv_file': csv_file,
            'record': record
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-csv-records/{csv_file}")
async def list_csv_records(csv_file: str):
    """List all CSV records."""
    try:
        records = csv_handler.list_all_records(csv_file)

        return JSONResponse({
            'status': 'success',
            'csv_file': csv_file,
            'total_records': len(records),
            'records': records
        })

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))









CSV_FOLDER = "CSV"

@app.get("/open-csv/{csv_file}")
async def open_csv(csv_file: str):
    """
    Open the CSV in browser (inline), no download.
    Reads from /CSV/ folder.
    """

    if csv_file not in ["thailand_data_bangkok", "thailand_data_phuket"]:
        raise HTTPException(status_code=400, detail="Invalid CSV name")

    csv_filename = "thailand_data_bangkok.csv" if csv_file == "thailand_data_bangkok" else "thailand_data_phuket.csv"
    csv_path = os.path.join(CSV_FOLDER, csv_filename)

    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found at {csv_path}")

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            csv_text = f.read()

        return Response(
            content=csv_text,
            media_type="text/csv"   # 🚀 Browser shows CSV directly
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV: {e}")



@app.get("/get-video-url/{ada_no}")
async def get_video_url(ada_no: str):
    """
    Return the S3 video URL for ada_no.
    Uses persistent caching to avoid repeated S3 calls.
    """
    # ---------- CACHE CHECK ----------
    if ada_no in video_cache:
        return {
            "status": "success",
            "cached": True,
            "ada_no": ada_no,
            "video_url": video_cache[ada_no]
        }

    # ---------- BUILD S3 KEY ----------
    s3_key = f"amway_videos/{ada_no}.mp4"

    # ---------- CHECK IF FILE EXISTS IN S3 ----------
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Video not found for ADA No: {ada_no}"
        )

    # ---------- BUILD PUBLIC URL ----------
    video_url = f"https://{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"

    # ---------- SAVE TO CACHE (RAM + FILE) ----------
    video_cache[ada_no] = video_url
    save_video_cache(video_cache)

    return {
        "status": "success",
        "cached": False,
        "ada_no": ada_no,
        "video_url": video_url
    }


@app.delete("/delete-today-uploads")
async def delete_today_uploads():
    """
    Delete all images uploaded today.
    Removes from S3, marks as deleted in metadata.
    """
    try:
        today = datetime.now().date().isoformat()  # "2024-12-07"
        
        # 1. Find all metadata entries uploaded today
        today_items = [
            (i, item) for i, item in enumerate(metadata)
            if item.get('uploaded_at', '').startswith(today)
        ]
        
        if not today_items:
            return {
                'status': 'no_data',
                'message': f"No images uploaded today ({today})",
                'total_deleted': 0
            }
        
        deleted_images = set()
        deleted_faces = 0
        affected_jobs = set()
        
        # 2. Delete from S3 and mark as deleted
        for idx, item in today_items:
            image_url = item['image_url']
            
            # Delete from S3 (only once per unique image)
            if image_url not in deleted_images:
                try:
                    s3_key = image_url.split(f"{S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/")[1]
                    s3_client.delete_object(Bucket=S3_BUCKET, Key=s3_key)
                    deleted_images.add(image_url)
                except Exception as e:
                    print(f"S3 delete failed for {image_url}: {e}")
            
            # Mark as deleted in metadata
            metadata[idx]['deleted'] = True
            metadata[idx]['deleted_at'] = datetime.now().isoformat()
            deleted_faces += 1
            
            # Track affected jobs
            if 'job_id' in item:
                affected_jobs.add(item['job_id'])
        
        # 3. Save metadata
        save_index()
        
        # 4. Update job statuses
        for job_id in affected_jobs:
            if job_id in jobs:
                jobs[job_id]['status'] = 'deleted'
                jobs[job_id]['deleted_at'] = datetime.now().isoformat()
        save_jobs()
        
        return {
            'status': 'success',
            'date': today,
            'total_faces_deleted': deleted_faces,
            'unique_images_deleted': len(deleted_images),
            'affected_jobs': len(affected_jobs),
            'deleted_images': list(deleted_images)[:10],  # Show first 10
            'note': 'Items marked as deleted. Run /rebuild-index to fully clean FAISS index.'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# ===================== MAIN =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
