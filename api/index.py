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
USERS_FILE = os.getenv('USERS_FILE', 'users.json')
APPOINTMENTS_FILE = os.getenv('APPOINTMENTS_FILE', 'appointments.json')
PATIENT_PROFILES_FILE = os.getenv('PATIENT_PROFILES_FILE', 'patient_profiles.json')
RECORDS_METADATA_FILE = os.getenv('RECORDS_METADATA_FILE', 'records_metadata.json')
NOTIFICATIONS_FILE = os.getenv('NOTIFICATIONS_FILE', 'notifications.json')
HEALTH_METRICS_FILE = os.getenv('HEALTH_METRICS_FILE', 'health_metrics.json')
HEALTH_ALERTS_FILE = os.getenv('HEALTH_ALERTS_FILE', 'health_alerts.json')

# Ensure required directories exist and initialize files
def initialize_directory_and_files():
    for dir_path in [UPLOAD_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    for file_path in [USERS_FILE, APPOINTMENTS_FILE, PATIENT_PROFILES_FILE, RECORDS_METADATA_FILE, NOTIFICATIONS_FILE, HEALTH_METRICS_FILE, HEALTH_ALERTS_FILE]:
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({}, f)

initialize_directory_and_files()

# Data persistence functions
def _load_data(file_path: str) -> dict:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def _save_data(file_path: str, data: dict) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

# Pydantic models for request/response validation
class Notification(BaseModel):
    notification_id: str
    user_id: str
    message: str
    created_at: datetime
    read: bool = False

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

# Authentication endpoints
@app.post("/api/auth/register")
async def register_user(email: str = Form(...), password: str = Form(...), role: str = Form(...)):
    try:
        users = _load_data(USERS_FILE)
        
        # Check if email is already registered
        if any(user.get("email") == email for user in users.values()):
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # In a real application, this would create a user in the database
        hashed_password = utils.hash_password(password)
        user_id = f"user_{uuid4().hex}"
        
        new_user = {
            "user_id": user_id,
            "email": email,
            "role": role,
            "password_hash": hashed_password,
            "name": "User Name Placeholder"
        }
        users[user_id] = new_user
        _save_data(USERS_FILE, users)
        
        return {
            "user_id": user_id,
            "email": email,
            "role": role,
            "message": "User registered successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/auth/login")
async def login(email: str = Form(...), password: str = Form(...)):
    try:
        users = _load_data(USERS_FILE)
        
        user_record = next((user for user in users.values() if user.get("email") == email), None)
        
        if not user_record or not utils.verify_password(password, user_record.get("password_hash")):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        token = utils.generate_jwt_token(user_record["user_id"], user_record["role"])
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user_record["user_id"],
            "role": user_record["role"]
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# Health record endpoints
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

        user = await get_current_user(token)
        if not validate_record_access(user, patient_id):
            raise HTTPException(status_code=403, detail="Access denied")

        record_id = f"record_{uuid4().hex}"
        file_extension = ALLOWED_FILE_TYPES[file.content_type]
        filename = f"{patient_id}_{record_id}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        log_phi_access(user["user_id"], "upload", f"Record uploaded for patient {patient_id}")

        records = _load_data(RECORDS_METADATA_FILE)
        records[record_id] = {
            "record_id": record_id,
            "patient_id": patient_id,
            "record_type": record_type,
            "upload_date": datetime.now(timezone.utc).isoformat(),
            "metadata": {
                "type": file.content_type,
                "size": file_size,
                "filename": filename
            },
            "access_granted_to": [] # Add this to match the Pydantic model
        }
        _save_data(RECORDS_METADATA_FILE, records)

        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "record_type": record_type
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/records/{patient_id}")
async def get_records(patient_id: str, token: str = Depends(oauth2_scheme)):
    try:
        user_data = utils.verify_jwt_token(token)
        if user_data["role"] == "patient" and user_data["user_id"] != patient_id:
            raise HTTPException(status_code=403, detail="Not authorized to view these records")
        
        records = _load_data(RECORDS_METADATA_FILE)
        patient_records = [
            record for record in records.values()
            if record["patient_id"] == patient_id
        ]
        return patient_records
    except Exception as e:
        raise HTTPException(status_code=404, detail="Records not found")

@app.get("/api/records/download/{record_id}")
async def download_record(record_id: str, token: str = Depends(oauth2_scheme)):
    try:
        user_data = utils.verify_jwt_token(token)
        records = _load_data(RECORDS_METADATA_FILE)
        
        if record_id not in records:
            raise HTTPException(status_code=404, detail="Record not found")
        
        file_data = records[record_id]
        file_path = os.path.join(UPLOAD_DIR, file_data["metadata"]["filename"])

        if user_data["role"] == "patient" and user_data["user_id"] != file_data["patient_id"]:
            raise HTTPException(status_code=403, detail="Not authorized to download this record")
        
        return FileResponse(
            path=file_path,
            filename=file_data["metadata"]["filename"],
            media_type=file_data["metadata"]["type"],
            headers={
                "Content-Disposition": f'attachment; filename="{file_data["metadata"]["filename"]}"'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Appointment endpoints
@app.post("/api/appointments")
async def create_appointment(
    patient_id: str = Form(...),
    date: datetime = Form(...),
    medical_condition: str = Form(...),
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
            "doctor_id": None,
            "date": date.isoformat(),
            "status": "pending",
            "medical_condition": medical_condition,
            "notes": notes,
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": None,
            "updated_by": None
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

# Add new search endpoint
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
        for user_id, user in users.items():
            if user["role"] == "patient":
                profile = patient_profiles.get(user_id, {})
                all_patients.append({
                    "user_id": user_id,
                    "name": user["name"],
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
        user_info = users.get(patient_id)
        
        if not user_info:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return {
            "user_id": patient_id,
            "name": user_info.get("name"),
            "email": user_info.get("email"),
            "profile": profile
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add new endpoint for doctor to schedule appointment
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
        notifications = _load_data(NOTIFICATIONS_FILE)
        
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
        
        notification_id = f"notif_{uuid4().hex}"
        notification = {
            "notification_id": notification_id,
            "user_id": patient_id,
            "message": f"New appointment scheduled with Dr. {user_data.get('name', 'N/A')} on {date.strftime('%Y-%m-%d %H:%M')}",
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

# Add endpoint to get notifications
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
        return [notif for notif in notifications.values() if notif["user_id"] == user_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Add endpoint to mark notification as read
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

# Add new endpoints for health data analysis
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

@app.get("/api/patients/{patient_id}/trends")
async def get_patient_trends(
    patient_id: str,
    metric_type: str,
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y
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

@app.post("/api/patients/{patient_id}/metrics")
async def add_patient_metric(
    patient_id: str,
    metric_type: str = Form(...),
    value: str = Form(...),  # Changed from float to str to handle blood pressure format
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