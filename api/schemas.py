"""
Pydantic schemas for Heart Disease API request and response data contracts.
"""
from typing import List

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years", json_schema_extra={"example": 63})
    sex: int = Field(..., ge=0, le=1, description="Sex (1 = male; 0 = female)", json_schema_extra={"example": 1})
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1=typical, 2=atypical, 3=non-anginal, 4=asymptomatic)", json_schema_extra={"example": 1})
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg", json_schema_extra={"example": 145})
    chol: int = Field(..., ge=80, le=600, description="Serum cholesterol in mg/dl", json_schema_extra={"example": 233})
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", json_schema_extra={"example": 1})
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0=normal, 1=ST-T wave abnormality, 2=left ventricular hypertrophy)", json_schema_extra={"example": 2})
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved", json_schema_extra={"example": 150})
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (1 = yes; 0 = no)", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", json_schema_extra={"example": 2.3})
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment (1=upsloping, 2=flat, 3=downsloping)", json_schema_extra={"example": 3})
    ca: int = Field(..., ge=0, le=3, description="Number of major vessels (0-3) colored by flourosopy", json_schema_extra={"example": 0})
    thal: int = Field(..., description="Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)", json_schema_extra={"example": 6})

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = Healthy, 1 = Heart Disease Risk")
    disease_present: bool = Field(..., description="True if heart disease risk is present")
    confidence: float = Field(..., description="Prediction probability confidence score [0.0 - 1.0]")
    probability_disease: float = Field(..., description="Estimated probability of heart disease")
    risk_level: str = Field(..., description="Clinical risk category: Low Risk, Moderate Risk, or High Risk")
    model_version: str = Field(default="0.1.0")

class BatchPredictionRequest(BaseModel):
    patients: List[PatientFeatures]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class ModelInfoResponse(BaseModel):
    model_name: str
    metrics: dict
    author: str
    dataset: str
