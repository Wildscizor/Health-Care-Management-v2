from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Query
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict
from datetime import datetime, timezone, timedelta
import json
from pydantic import BaseModel, validator
import utils
from fastapi.responses import FileResponse
import os
import base64
from utils import (
    verify_password, create_access_token, get_current_user,
    encrypt_sensitive_data, decrypt_sensitive_data, log_phi_access,
    validate_record_access, mask_sensitive_data,
    SecurityManager,
    HIPAACompliance,
    AccessControl
)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import traceback

# Load environment variables
load_dotenv()

app = FastAPI(title="Patient Health Records API")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Add CORS middleware with specific origins
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'http://localhost:8501').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Initialize security components
security_manager = SecurityManager()
hipaa = HIPAACompliance()
access_control = AccessControl()

# Constants for file upload
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "image/jpeg": ".jpg",
    "image/png": ".png"
}

"""Directories and files for persistent storage"""
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'uploads')
PROFILE_PICS_DIR = os.getenv('PROFILE_PICS_DIR', 'profile_pics')
BLOG_IMAGES_DIR = os.getenv('BLOG_IMAGES_DIR', 'blog_images')
USERS_FILE = os.getenv('USERS_FILE', 'users.json')
APPOINTMENTS_FILE = os.getenv('APPOINTMENTS_FILE', 'appointments.json')
PATIENT_PROFILES_FILE = os.getenv('PATIENT_PROFILES_FILE', 'patient_profiles.json')
NOTIFICATIONS_FILE = os.getenv('NOTIFICATIONS_FILE', 'notifications.json')
HEALTH_METRICS_FILE = os.getenv('HEALTH_METRICS_FILE', 'health_metrics.json')
HEALTH_ALERTS_FILE = os.getenv('HEALTH_ALERTS_FILE', 'health_alerts.json')

# Ensure required directories/files exist and initialize them if they don't
for dir_path in [UPLOAD_DIR, PROFILE_PICS_DIR, BLOG_IMAGES_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def initialize_json_file(file_path, default_content={}):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(default_content, f)

initialize_json_file(USERS_FILE, {})
initialize_json_file(APPOINTMENTS_FILE, {})
initialize_json_file(PATIENT_PROFILES_FILE, {})
initialize_json_file(NOTIFICATIONS_FILE, {})
initialize_json_file(HEALTH_METRICS_FILE, {})
initialize_json_file(HEALTH_ALERTS_FILE, {})

# Serve profile pictures and blog images as static files
app.mount("/profile_pics", StaticFiles(directory=PROFILE_PICS_DIR), name="profile_pics")
app.mount("/blog_images", StaticFiles(directory=BLOG_IMAGES_DIR), name="blog_images")

# -----------------------------
# Database (MySQL via SQLAlchemy)
# -----------------------------
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DB = os.getenv('MYSQL_DB', 'healthcare')
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')

DATABASE_URL = os.getenv(
    'DATABASE_URL',
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BlogPost(Base):
    __tablename__ = 'blog_posts'

    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(String(64), index=True, nullable=False)
    title = Column(String(255), nullable=False)
    image_path = Column(String(512), nullable=True)  # served via /blog_images
    category = Column(String(64), index=True, nullable=False)
    summary = Column(String(1024), nullable=False)
    content = Column(Text, nullable=False)
    is_draft = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=datetime.now(timezone.utc))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def on_startup_create_tables():
    try:
        Base.metadata.create_all(bind=engine)
    except SQLAlchemyError as e:
        print("[startup] DB init error:", str(e))

# Pydantic models for request/response validation
class User(BaseModel):
    user_id: str
    role: str
    email: str
    name: str

class HealthRecord(BaseModel):
    record_id: str
    patient_id: str
    record_type: str
    upload_date: datetime
    metadata: dict
    access_granted_to: List[str]

class Appointment(BaseModel):
    appointment_id: str
    patient_id: str
    doctor_id: str
    date: datetime
    status: str
    notes: Optional[str]

# Add new models for notifications
class Notification(BaseModel):
    notification_id: str
    user_id: str
    message: str
    created_at: datetime
    read: bool = False

# Add new models for health data
class HealthMetric(BaseModel):
    metric_id: str
    patient_id: str
    metric_type: str
    value: float
    unit: str
    timestamp: datetime
    is_critical: bool = False

    @validator('metric_type')
    def validate_metric_type(cls, v):
        valid_types = ['blood_pressure', 'heart_rate', 'blood_sugar', 'temperature', 'weight']
        if v not in valid_types:
            raise ValueError(f'metric_type must be one of {valid_types}')
        return v

    @validator('value')
    def validate_value(cls, v, values):
        if 'metric_type' in values:
            if values['metric_type'] == 'blood_pressure':
                if not isinstance(v, (int, float)) or v < 0 or v > 300:
                    raise ValueError('Blood pressure must be between 0 and 300')
            elif values['metric_type'] == 'heart_rate':
                if not isinstance(v, (int, float)) or v < 0 or v > 250:
                    raise ValueError('Heart rate must be between 0 and 250')
            elif values['metric_type'] == 'blood_sugar':
                if not isinstance(v, (int, float)) or v < 0 or v > 1000:
                    raise ValueError('Blood sugar must be between 0 and 1000')
        return v

class HealthAlert(BaseModel):
    alert_id: str
    patient_id: str
    metric_type: str
    value: float
    threshold: float
    severity: str  # "critical", "warning", "info"
    timestamp: datetime
    message: str
    is_resolved: bool = False

# -----------------------------
# Blog Pydantic Models
# -----------------------------
class BlogCreate(BaseModel):
    title: str
    category: str
    summary: str
    content: str
    is_draft: bool = False

class BlogOut(BaseModel):
    id: int
    author_id: str
    title: str
    image_url: Optional[str]
    category: str
    summary: str
    is_draft: bool
    created_at: datetime
    class Config:
        from_attributes = True

# -----------------------------
# Data Persistence Functions
# -----------------------------

def _load_data(file_path: str) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception:
        return {}

def _save_data(file_path: str, data: dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def _user_public_view(user: dict) -> dict:
    return {
        "user_id": user["user_id"],
        "role": user["role"],
        "email": user["email"],
        "username": user["username"],
        "first_name": user.get("first_name"),
        "last_name": user.get("last_name"),
        "address": user.get("address"),
        "profile_picture": user.get("profile_picture"),
    }

# -----------------------------
# Blog Endpoints
# -----------------------------
ALLOWED_BLOG_CATEGORIES = {"Mental Health", "Heart Disease", "Covid19", "Immunization"}

@app.post("/api/blogs")
async def create_blog(
    title: str = Form(...),
    category: str = Form(...),
    summary: str = Form(...),
    content: str = Form(...),
    is_draft: bool = Form(False),
    image: UploadFile = File(None),
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
):
    try:
        user = utils.verify_jwt_token(token)
        if user["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can create blog posts")

        if category not in ALLOWED_BLOG_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Allowed: {', '.join(sorted(ALLOWED_BLOG_CATEGORIES))}")

        image_path = None
        if image is not None:
            ext = os.path.splitext(image.filename)[1] or ".jpg"
            safe_name = f"blog_{uuid4().hex}{ext}"
            fs_path = os.path.join(BLOG_IMAGES_DIR, safe_name)
            content_bytes = await image.read()
            with open(fs_path, 'wb') as f:
                f.write(content_bytes)
            image_path = f"blog_images/{safe_name}"

        post = BlogPost(
            author_id=user["user_id"],
            title=title,
            image_path=image_path,
            category=category,
            summary=summary,
            content=content,
            is_draft=is_draft,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(post)
        db.commit()
        db.refresh(post)

        return {
            "id": post.id,
            "message": "Blog post created successfully"
        }
    except HTTPException as he:
        raise he
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/blogs", response_model=List[BlogOut])
async def list_blogs(
    category: Optional[str] = Query(None),
    include_drafts: bool = Query(False),
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
):
    try:
        user = utils.verify_jwt_token(token)
        q = db.query(BlogPost)
        if category:
            if category not in ALLOWED_BLOG_CATEGORIES:
                raise HTTPException(status_code=400, detail="Invalid category")
            q = q.filter(BlogPost.category == category)
        # Honor include_drafts for all roles so patient portal can opt-in to show all
        if not include_drafts:
            q = q.filter(BlogPost.is_draft == False)
        posts = q.order_by(BlogPost.created_at.desc()).all()
        results = []
        for p in posts:
            results.append({
                "id": p.id,
                "author_id": p.author_id,
                "title": p.title,
                "image_url": (f"/" + p.image_path) if p.image_path else None,
                "category": p.category,
                "summary": p.summary,
                "is_draft": p.is_draft,
                "created_at": p.created_at
            })
        return results
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/blogs/mine", response_model=List[BlogOut])
async def list_my_blogs(
    token: str = Depends(oauth2_scheme),
    db = Depends(get_db)
):
    try:
        user = utils.verify_jwt_token(token)
        if user["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view their posts")
        posts = db.query(BlogPost).filter(BlogPost.author_id == user["user_id"]).order_by(BlogPost.created_at.desc()).all()
        return [
            {
                "id": p.id,
                "author_id": p.author_id,
                "title": p.title,
                "image_url": (f"/" + p.image_path) if p.image_path else None,
                "category": p.category,
                "summary": p.summary,
                "is_draft": p.is_draft,
                "created_at": p.created_at
            }
            for p in posts
        ]
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Authentication Endpoints
# -----------------------------

@app.post("/api/auth/signup")
async def signup(
    first_name: str = Form(...),
    last_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    role: str = Form(...),  # "patient" or "doctor"
    address_line1: str = Form(...),
    city: str = Form(...),
    state: str = Form(...),
    pincode: str = Form(...),
    profile_picture: UploadFile = File(None)
):
    try:
        if role not in ["patient", "doctor"]:
            raise HTTPException(status_code=400, detail="Invalid role. Choose 'patient' or 'doctor'")
        if password != confirm_password:
            raise HTTPException(status_code=400, detail="Password and confirm password do not match")

        users = _load_data(USERS_FILE)
        if username in users:
            raise HTTPException(status_code=400, detail="Username already exists")
        if any(u.get("email") == email for u in users.values()):
            raise HTTPException(status_code=400, detail="Email already registered")

        user_id = f"user_{uuid4().hex}"

        picture_path = None
        if profile_picture is not None:
            pic_ext = os.path.splitext(profile_picture.filename)[1] or ".jpg"
            safe_name = f"{user_id}{pic_ext}"
            picture_fs_path = os.path.join(PROFILE_PICS_DIR, safe_name)
            content = await profile_picture.read()
            with open(picture_fs_path, "wb") as pf:
                pf.write(content)
            picture_path = f"profile_pics/{safe_name}"

        address = {
            "line1": address_line1,
            "city": city,
            "state": state,
            "pincode": pincode
        }

        users[username] = {
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password_hash": utils.get_password_hash(password),
            "role": role,
            "address": address,
            "profile_picture": picture_path
        }
        _save_data(USERS_FILE, users)

        return {
            "message": "Signup successful",
            "user": _user_public_view(users[username])
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(email: str = Form(None), username: str = Form(None), password: str = Form(...)):
    try:
        users = _load_data(USERS_FILE)

        user_record = None
        if username and username in users:
            user_record = users[username]
        elif email:
            for u in users.values():
                if u.get("email") == email:
                    user_record = u
                    break

        if not user_record:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        if not utils.verify_password(password, user_record["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        token = utils.generate_jwt_token(user_record["user_id"], user_record["role"])
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_record["user_id"],
            "role": user_record["role"]
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@app.get("/api/auth/user/{user_id}")
async def get_user(user_id: str, token: str = Depends(oauth2_scheme)):
    try:
        utils.verify_jwt_token(token)
        users = _load_data(USERS_FILE)
        for u in users.values():
            if u["user_id"] == user_id:
                return _user_public_view(u)
        raise HTTPException(status_code=404, detail="User not found")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Health Record Endpoints
# -----------------------------

@app.post("/api/records/upload")
async def upload_record(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
    record_type: str = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        if file.content_type not in ALLOWED_FILE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES.keys())}"
            )

        file_size = 0
        chunk_size = 1024 * 1024
        while chunk := await file.read(chunk_size):
            file_size += len(chunk)
            if file_size > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=400,
                    detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE/1024/1024}MB"
                )
        await file.seek(0)

        user = utils.verify_jwt_token(token)
        if user["role"] == "patient" and user["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Patients can only upload their own records")
        if not validate_record_access(patient_id, user["user_id"], None):
            raise HTTPException(status_code=403, detail="Access denied")

        record_id = f"record_{uuid4().hex}"
        file_extension = ALLOWED_FILE_TYPES[file.content_type]
        filename = f"{patient_id}_{record_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        log_phi_access(user["user_id"], "upload", f"Record uploaded for patient {patient_id}")

        # Store record metadata
        records_metadata = _load_data(os.path.join(UPLOAD_DIR, 'records_metadata.json'))
        records_metadata[record_id] = {
            "record_id": record_id,
            "patient_id": patient_id,
            "record_type": record_type,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "type": file.content_type,
                "filename": filename,
                "size": file_size,
            },
        }
        _save_data(os.path.join(UPLOAD_DIR, 'records_metadata.json'), records_metadata)

        return {
            "message": "File uploaded successfully",
            "record_id": record_id
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print("[upload_record] Unexpected error:", str(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

@app.get("/api/records/{patient_id}")
async def get_records(patient_id: str, token: str = Depends(oauth2_scheme)):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] == "patient" and user_data["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Not authorized to view these records")
        
        records_metadata = _load_data(os.path.join(UPLOAD_DIR, 'records_metadata.json'))
        records = [
            record for record in records_metadata.values()
            if record["patient_id"] == patient_id
        ]

        return records
    except Exception as e:
        raise HTTPException(status_code=404, detail="Records not found")

@app.get("/api/records/download/{record_id}")
async def download_record(record_id: str, token: str = Depends(oauth2_scheme)):
    try:
        user_data = utils.verify_jwt_token(token)
        records_metadata = _load_data(os.path.join(UPLOAD_DIR, 'records_metadata.json'))

        if record_id not in records_metadata:
            raise HTTPException(status_code=404, detail="Record not found")
        
        record_data = records_metadata[record_id]
        file_path = os.path.join(UPLOAD_DIR, record_data["metadata"]["filename"])

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on server")

        # Security check: ensure the user has permission to download this file
        if user_data["role"] == "patient" and user_data["user_id"] != record_data["patient_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to download this file")

        log_phi_access(user_data["user_id"], "download", f"Record downloaded for patient {record_data['patient_id']}")
        
        return FileResponse(
            path=file_path,
            filename=record_data["metadata"]["filename"],
            media_type=record_data["metadata"]["type"],
            headers={
                "Content-Disposition": f'attachment; filename="{record_data["metadata"]["filename"]}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Appointment Endpoints
# -----------------------------

@app.post("/api/appointments")
async def create_appointment(
    patient_id: str = Form(...),
    date: datetime = Form(...),
    notes: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        
        if user_data["role"] != "patient" or user_data["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Only patients can create their own appointments")
        
        appointment_id = f"appt_{uuid4().hex}"
        appointments = _load_data(APPOINTMENTS_FILE)
        
        appointment = {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "doctor_id": None, # Doctor will be assigned later
            "date": date.isoformat(),
            "status": "pending",
            "notes": notes,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        
        appointments[appointment_id] = appointment
        _save_data(APPOINTMENTS_FILE, appointments)
        
        return appointment
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/appointments")
async def get_appointments(
    user_id: str,
    role: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        appointments = _load_data(APPOINTMENTS_FILE)
        
        if role == "doctor":
            return [apt for apt in appointments.values() if apt.get("doctor_id") == user_id]
        else:
            return [apt for apt in appointments.values() if apt.get("patient_id") == user_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/appointments/{appointment_id}")
async def update_appointment_status(
    appointment_id: str,
    status: dict,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can update appointment status")
        
        appointments = _load_data(APPOINTMENTS_FILE)
        if appointment_id not in appointments:
            raise HTTPException(status_code=404, detail="Appointment not found")
        
        appointments[appointment_id]["status"] = status["status"]
        appointments[appointment_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
        appointments[appointment_id]["updated_by"] = user_data["user_id"]
        _save_data(APPOINTMENTS_FILE, appointments)
        
        return {
            "appointment_id": appointment_id,
            "status": status["status"],
            "updated_at": appointments[appointment_id]["updated_at"],
            "message": "Appointment status updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/appointments/schedule")
async def schedule_appointment(
    patient_id: str = Form(...),
    date: datetime = Form(...),
    notes: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can schedule appointments")
        
        appointment_id = f"appt_{uuid4().hex}"
        appointments = _load_data(APPOINTMENTS_FILE)
        
        appointment = {
            "appointment_id": appointment_id,
            "patient_id": patient_id,
            "doctor_id": user_data["user_id"],
            "date": date.isoformat(),
            "status": "scheduled",
            "notes": notes,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "updated_by": user_data["user_id"]
        }
        
        appointments[appointment_id] = appointment
        _save_data(APPOINTMENTS_FILE, appointments)
        
        notifications = _load_data(NOTIFICATIONS_FILE)
        notification_id = f"notif_{uuid4().hex}"
        notification = {
            "notification_id": notification_id,
            "user_id": patient_id,
            "message": f"New appointment scheduled with Dr. {user_data['first_name']} {user_data['last_name']} on {date.strftime('%Y-%m-%d %H:%M')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "read": False
        }
        notifications[notification_id] = notification
        _save_data(NOTIFICATIONS_FILE, notifications)
        
        return {
            "appointment": appointment,
            "notification": notification
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -----------------------------
# Patient Profile & Search Endpoints
# -----------------------------

@app.get("/api/patients/search")
async def search_patients(
    query: str = None,
    condition: str = None,
    age_min: int = None,
    age_max: int = None,
    gender: str = None,
    blood_type: str = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can search patients")
        
        patient_profiles = _load_data(PATIENT_PROFILES_FILE)
        users = _load_data(USERS_FILE)
        
        all_patients = []
        for user_key, user in users.items():
            if user["role"] == "patient":
                profile = patient_profiles.get(user["user_id"], {})
                all_patients.append({
                    "user_id": user["user_id"],
                    "name": f"{user['first_name']} {user['last_name']}",
                    "email": user["email"],
                    "profile": profile
                })
        
        filtered_patients = all_patients
        
        if query:
            filtered_patients = [
                p for p in filtered_patients
                if query.lower() in p["name"].lower() or
                   query.lower() in p["email"].lower() or
                   any(query.lower() in cond.lower() for cond in p["profile"].get("medical_conditions", []))
            ]
        
        if condition:
            filtered_patients = [
                p for p in filtered_patients
                if condition.lower() in [c.lower() for c in p["profile"].get("medical_conditions", [])]
            ]
        
        if age_min is not None:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"].get("age", 0) >= age_min
            ]
        
        if age_max is not None:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"].get("age", 0) <= age_max
            ]
        
        if gender:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"].get("gender", "").lower() == gender.lower()
            ]
        
        if blood_type:
            filtered_patients = [
                p for p in filtered_patients
                if p["profile"].get("blood_type", "").lower() == blood_type.lower()
            ]
        
        return filtered_patients
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/profile")
async def get_patient_profile(
    patient_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient profiles")
        
        patient_profiles = _load_data(PATIENT_PROFILES_FILE)
        profile = patient_profiles.get(patient_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Patient profile not found")
        
        users = _load_data(USERS_FILE)
        user_info = next((user for user in users.values() if user["user_id"] == patient_id), None)
        
        if not user_info:
            raise HTTPException(status_code=404, detail="Patient user not found")
        
        return {
            "user_id": patient_id,
            "name": f"{user_info['first_name']} {user_info['last_name']}",
            "email": user_info['email'],
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Notifications Endpoints
# -----------------------------

@app.get("/api/notifications")
async def get_notifications(
    user_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Not authorized to view these notifications")
        
        notifications = _load_data(NOTIFICATIONS_FILE)
        user_notifications = [
            notif for notif in notifications.values()
            if notif["user_id"] == user_id
        ]
        
        return user_notifications
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/notifications/{notification_id}")
async def mark_notification_read(
    notification_id: str,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        notifications = _load_data(NOTIFICATIONS_FILE)
        
        if notification_id not in notifications:
            raise HTTPException(status_code=404, detail="Notification not found")
        
        if notifications[notification_id]["user_id"] != user_data["user_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to modify this notification")
        
        notifications[notification_id]["read"] = True
        _save_data(NOTIFICATIONS_FILE, notifications)
        
        return {"message": "Notification marked as read"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------------
# Health Data Analysis Endpoints
# -----------------------------

@app.get("/api/patients/{patient_id}/metrics")
async def get_patient_metrics(
    patient_id: str,
    metric_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient metrics")
        
        metrics = _load_data(HEALTH_METRICS_FILE)
        patient_metrics = [
            metric for metric in metrics.values()
            if metric["patient_id"] == patient_id
        ]
        
        if metric_type:
            patient_metrics = [m for m in patient_metrics if m["metric_type"] == metric_type]
        if start_date:
            patient_metrics = [m for m in patient_metrics if datetime.fromisoformat(m["timestamp"]) >= start_date]
        if end_date:
            patient_metrics = [m for m in patient_metrics if datetime.fromisoformat(m["timestamp"]) <= end_date]
        
        return patient_metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/alerts")
async def get_patient_alerts(
    patient_id: str,
    severity: Optional[str] = None,
    resolved: Optional[bool] = None,
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient alerts")
        
        alerts = _load_data(HEALTH_ALERTS_FILE)
        patient_alerts = [
            alert for alert in alerts.values()
            if alert["patient_id"] == patient_id
        ]
        
        if severity:
            patient_alerts = [a for a in patient_alerts if a["severity"] == severity]
        if resolved is not None:
            patient_alerts = [a for a in patient_alerts if a["is_resolved"] == resolved]
        
        return patient_alerts
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/patients/{patient_id}/metrics")
async def add_patient_metric(
    patient_id: str,
    metric_type: str = Form(...),
    value: str = Form(...),
    unit: str = Form(...),
    timestamp: datetime = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        
        metric_id = f"metric_{uuid4().hex}"
        metrics = _load_data(HEALTH_METRICS_FILE)
        
        is_critical = False
        if metric_type == "blood_pressure":
            systolic, diastolic = map(float, value.split("/"))
            if systolic > 140 or diastolic > 90:
                is_critical = True
        else:
            value_float = float(value)
            if metric_type == "heart_rate":
                if value_float > 100 or value_float < 60:
                    is_critical = True
            elif metric_type == "blood_sugar":
                if value_float > 140 or value_float < 70:
                    is_critical = True
        
        metric = {
            "metric_id": metric_id,
            "patient_id": patient_id,
            "metric_type": metric_type,
            "value": value,
            "unit": unit,
            "timestamp": timestamp.isoformat(),
            "is_critical": is_critical
        }
        
        metrics[metric_id] = metric
        _save_data(HEALTH_METRICS_FILE, metrics)
        
        if is_critical:
            alerts = _load_data(HEALTH_ALERTS_FILE)
            alert_id = f"alert_{uuid4().hex}"
            alert = {
                "alert_id": alert_id,
                "patient_id": patient_id,
                "metric_type": metric_type,
                "value": value,
                "threshold": "140/90" if metric_type == "blood_pressure" else (100 if metric_type == "heart_rate" else 140),
                "severity": "critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": f"Critical {metric_type} value: {value} {unit}",
                "is_resolved": False
            }
            alerts[alert_id] = alert
            _save_data(HEALTH_ALERTS_FILE, alerts)
        
        return metric
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/patients/{patient_id}/trends")
async def get_patient_trends(
    patient_id: str,
    metric_type: str,
    period: str = "1m",
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can view patient trends")
        
        end_date = datetime.now(timezone.utc)
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            raise HTTPException(status_code=400, detail="Invalid period")
        
        metrics = _load_data(HEALTH_METRICS_FILE)
        patient_metrics = [
            metric for metric in metrics.values()
            if metric["patient_id"] == patient_id
            and metric["metric_type"] == metric_type
            and datetime.fromisoformat(metric["timestamp"]) >= start_date
            and datetime.fromisoformat(metric["timestamp"]) <= end_date
        ]
        
        if patient_metrics:
            values = [m["value"] for m in patient_metrics]
            stats = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "latest": values[-1],
                "trend": "increasing" if values[-1] > values[0] else "decreasing",
                "data_points": len(values)
            }
        else:
            stats = {
                "min": None,
                "max": None,
                "avg": None,
                "latest": None,
                "trend": None,
                "data_points": 0
            }
        
        return {
            "metric_type": metric_type,
            "period": period,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "statistics": stats,
            "metrics": patient_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/alerts/{alert_id}")
async def update_alert_status(
    alert_id: str,
    is_resolved: bool = Form(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] != "doctor":
            raise HTTPException(status_code=403, detail="Only doctors can update alerts")
        
        alerts = _load_data(HEALTH_ALERTS_FILE)
        if alert_id not in alerts:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alerts[alert_id]["is_resolved"] = is_resolved
        _save_data(HEALTH_ALERTS_FILE, alerts)
        
        return {"message": "Alert status updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))