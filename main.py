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

load_dotenv()

app = FastAPI()

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

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            all_files = [f for f in zip_ref.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            for idx, filename in enumerate(all_files):
                try:
                    with zip_ref.open(filename) as img_file:
                        image_bytes = img_file.read()

                    # process image embeddings, add to FAISS, etc.

                except Exception as e:
                    # handle failed images

                    update_job_status(job_id, JobStatus.PROCESSING, progress=int((idx+1)/len(all_files)*100))

        update_job_status(job_id, JobStatus.COMPLETED)

    finally:
        os.remove(zip_path)  # cleanup disk



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


@app.post("/search-face")
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


# ===================== MAIN =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
