# Patient-Level MI Diagnosis Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the patient-level MI diagnosis analysis system into existing healthcare applications, research pipelines, and clinical decision support systems.

## Quick Start Integration

### 1. Basic Integration

```python
from lighten_ml.pipeline import PatientLevelClinicalPipeline
from lighten_ml.data_loaders import PatientHistoryLoader, LabDataLoader

# Initialize the pipeline
def create_mi_analysis_pipeline(discharge_notes_path, lab_events_path, lab_items_path):
    """Create and configure the patient-level MI analysis pipeline."""
    
    # Initialize data loaders
    patient_loader = PatientHistoryLoader(discharge_notes_path)
    lab_loader = LabDataLoader(lab_events_path, lab_items_path)
    
    # Create pipeline
    pipeline = PatientLevelClinicalPipeline(
        patient_history_loader=patient_loader,
        lab_data_loader=lab_loader
    )
    
    return pipeline

# Usage
pipeline = create_mi_analysis_pipeline(
    "data/discharge_notes.csv",
    "data/labevents.csv", 
    "data/d_labitems.csv"
)

# Analyze a patient
result = pipeline.process_patient("12345")
```

### 2. Web API Integration

```python
from flask import Flask, request, jsonify
from lighten_ml.pipeline import PatientLevelClinicalPipeline

app = Flask(__name__)

# Initialize pipeline once at startup
pipeline = create_mi_analysis_pipeline(
    "data/discharge_notes.csv",
    "data/labevents.csv",
    "data/d_labitems.csv"
)

@app.route('/api/mi-analysis/<patient_id>', methods=['GET'])
def analyze_patient_mi(patient_id):
    """API endpoint for patient-level MI analysis."""
    try:
        result = pipeline.process_patient(patient_id)
        return jsonify({
            "status": "success",
            "patient_id": patient_id,
            "mi_diagnosis": result["mi_diagnosis"],
            "onset_date": result["onset_date"],
            "evidence_summary": result["evidence_summary"],
            "metadata": result["metadata"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "patient_id": patient_id,
            "error": str(e)
        }), 500

@app.route('/api/batch-analysis', methods=['POST'])
def batch_analyze_patients():
    """API endpoint for batch patient analysis."""
    try:
        patient_ids = request.json.get('patient_ids', [])
        results = []
        
        for patient_id in patient_ids:
            try:
                result = pipeline.process_patient(patient_id)
                results.append({
                    "patient_id": patient_id,
                    "status": "success",
                    "mi_diagnosis": result["mi_diagnosis"],
                    "onset_date": result["onset_date"]
                })
            except Exception as e:
                results.append({
                    "patient_id": patient_id,
                    "status": "error",
                    "error": str(e)
                })
        
        return jsonify({
            "status": "success",
            "total_patients": len(patient_ids),
            "results": results
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
```

## Advanced Integration Patterns

### 1. Database Integration

```python
import sqlite3
from datetime import datetime
from lighten_ml.pipeline import PatientLevelClinicalPipeline

class MIAnalysisService:
    """Service class for MI analysis with database integration."""
    
    def __init__(self, pipeline, db_path="mi_analysis.db"):
        self.pipeline = pipeline
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the results database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mi_analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mi_diagnosis TEXT NOT NULL,
                onset_date TEXT,
                evidence_summary TEXT,
                metadata TEXT,
                processing_time_seconds REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def analyze_and_store(self, patient_id):
        """Analyze patient and store results in database."""
        import time
        import json
        
        start_time = time.time()
        
        try:
            # Perform analysis
            result = self.pipeline.process_patient(patient_id)
            processing_time = time.time() - start_time
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mi_analysis_results 
                (patient_id, mi_diagnosis, onset_date, evidence_summary, metadata, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                result["mi_diagnosis"],
                result["onset_date"],
                json.dumps(result["evidence_summary"]),
                json.dumps(result["metadata"]),
                processing_time
            ))
            
            conn.commit()
            conn.close()
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Store error in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO mi_analysis_results 
                (patient_id, mi_diagnosis, onset_date, evidence_summary, metadata, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                patient_id,
                "ERROR",
                None,
                json.dumps({"error": str(e)}),
                json.dumps({"error": True}),
                processing_time
            ))
            
            conn.commit()
            conn.close()
            
            raise
    
    def get_patient_history(self, patient_id):
        """Get analysis history for a patient."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM mi_analysis_results 
            WHERE patient_id = ? 
            ORDER BY analysis_date DESC
        ''', (patient_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
```

### 2. Async Processing Integration

```python
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from lighten_ml.pipeline import PatientLevelClinicalPipeline

class AsyncMIAnalysisService:
    """Async service for high-throughput MI analysis."""
    
    def __init__(self, pipeline, max_workers=4):
        self.pipeline = pipeline
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def analyze_patient_async(self, patient_id):
        """Async wrapper for patient analysis."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self.pipeline.process_patient, 
            patient_id
        )
    
    async def batch_analyze_async(self, patient_ids, batch_size=10):
        """Async batch processing with concurrency control."""
        results = []
        
        # Process in batches to control memory usage
        for i in range(0, len(patient_ids), batch_size):
            batch = patient_ids[i:i + batch_size]
            
            # Create tasks for concurrent processing
            tasks = [
                self.analyze_patient_async(patient_id) 
                for patient_id in batch
            ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for patient_id, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results.append({
                        "patient_id": patient_id,
                        "status": "error",
                        "error": str(result)
                    })
                else:
                    results.append({
                        "patient_id": patient_id,
                        "status": "success",
                        "result": result
                    })
        
        return results

# Usage
async def main():
    pipeline = create_mi_analysis_pipeline(
        "data/discharge_notes.csv",
        "data/labevents.csv",
        "data/d_labitems.csv"
    )
    
    service = AsyncMIAnalysisService(pipeline)
    patient_ids = ["12345", "67890", "11111"]
    
    results = await service.batch_analyze_async(patient_ids)
    print(f"Processed {len(results)} patients")

# Run async processing
# asyncio.run(main())
```

### 3. Microservice Integration

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid
from lighten_ml.pipeline import PatientLevelClinicalPipeline

app = FastAPI(title="MI Analysis Microservice", version="1.0.0")

# Global pipeline instance
pipeline = None

# Request/Response models
class AnalysisRequest(BaseModel):
    patient_id: str
    priority: Optional[str] = "normal"

class BatchAnalysisRequest(BaseModel):
    patient_ids: List[str]
    callback_url: Optional[str] = None

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    patient_id: str
    mi_diagnosis: Optional[str] = None
    onset_date: Optional[str] = None
    evidence_summary: Optional[dict] = None
    error: Optional[str] = None

# In-memory job tracking (use Redis in production)
jobs = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    pipeline = create_mi_analysis_pipeline(
        "data/discharge_notes.csv",
        "data/labevents.csv",
        "data/d_labitems.csv"
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_patient(request: AnalysisRequest):
    """Synchronous patient analysis endpoint."""
    job_id = str(uuid.uuid4())
    
    try:
        result = pipeline.process_patient(request.patient_id)
        
        response = AnalysisResponse(
            job_id=job_id,
            status="completed",
            patient_id=request.patient_id,
            mi_diagnosis=result["mi_diagnosis"],
            onset_date=result["onset_date"],
            evidence_summary=result["evidence_summary"]
        )
        
        jobs[job_id] = response
        return response
        
    except Exception as e:
        response = AnalysisResponse(
            job_id=job_id,
            status="failed",
            patient_id=request.patient_id,
            error=str(e)
        )
        
        jobs[job_id] = response
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def batch_analyze(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Asynchronous batch analysis endpoint."""
    job_id = str(uuid.uuid4())
    
    # Start background processing
    background_tasks.add_task(
        process_batch_analysis, 
        job_id, 
        request.patient_ids, 
        request.callback_url
    )
    
    return {"job_id": job_id, "status": "processing", "patient_count": len(request.patient_ids)}

async def process_batch_analysis(job_id: str, patient_ids: List[str], callback_url: Optional[str]):
    """Background task for batch processing."""
    results = []
    
    for patient_id in patient_ids:
        try:
            result = pipeline.process_patient(patient_id)
            results.append({
                "patient_id": patient_id,
                "status": "success",
                "mi_diagnosis": result["mi_diagnosis"],
                "onset_date": result["onset_date"]
            })
        except Exception as e:
            results.append({
                "patient_id": patient_id,
                "status": "error",
                "error": str(e)
            })
    
    # Store results
    jobs[job_id] = {
        "job_id": job_id,
        "status": "completed",
        "results": results
    }
    
    # Optional callback notification
    if callback_url:
        # Implement callback notification logic here
        pass

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "MI Analysis"}
```

## Configuration Management

### 1. Environment-Based Configuration

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class MIAnalysisConfig:
    """Configuration class for MI analysis system."""
    
    # Data paths
    discharge_notes_path: str = os.getenv("DISCHARGE_NOTES_PATH", "data/discharge_notes.csv")
    lab_events_path: str = os.getenv("LAB_EVENTS_PATH", "data/labevents.csv")
    lab_items_path: str = os.getenv("LAB_ITEMS_PATH", "data/d_labitems.csv")
    
    # Analysis parameters
    troponin_threshold: float = float(os.getenv("TROPONIN_THRESHOLD", "0.04"))
    require_both_criteria: bool = os.getenv("REQUIRE_BOTH_CRITERIA", "true").lower() == "true"
    
    # Performance settings
    max_workers: int = int(os.getenv("MAX_WORKERS", "4"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE")
    
    # Database
    database_url: Optional[str] = os.getenv("DATABASE_URL")
    
    # LLM settings
    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

def create_configured_pipeline(config: MIAnalysisConfig):
    """Create pipeline with configuration."""
    from lighten_ml.rule_engines import MIRuleEngineConfig
    
    # Create rule engine config
    rule_config = MIRuleEngineConfig(
        troponin_threshold=config.troponin_threshold,
        require_both_criteria=config.require_both_criteria
    )
    
    # Create pipeline
    pipeline = create_mi_analysis_pipeline(
        config.discharge_notes_path,
        config.lab_events_path,
        config.lab_items_path
    )
    
    return pipeline
```

### 2. Docker Integration

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  mi-analysis:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DISCHARGE_NOTES_PATH=/data/discharge_notes.csv
      - LAB_EVENTS_PATH=/data/labevents.csv
      - LAB_ITEMS_PATH=/data/d_labitems.csv
      - LOG_LEVEL=INFO
      - MAX_WORKERS=4
    volumes:
      - ./data:/data
      - ./logs:/app/logs
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
```

## Testing Integration

### 1. Unit Testing

```python
import unittest
from unittest.mock import Mock, patch
from lighten_ml.pipeline import PatientLevelClinicalPipeline

class TestMIAnalysisIntegration(unittest.TestCase):
    """Test cases for MI analysis integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_patient_loader = Mock()
        self.mock_lab_loader = Mock()
        
        self.pipeline = PatientLevelClinicalPipeline(
            patient_history_loader=self.mock_patient_loader,
            lab_data_loader=self.mock_lab_loader
        )
    
    def test_successful_analysis(self):
        """Test successful patient analysis."""
        # Mock data
        self.mock_patient_loader.get_patient_visit_history.return_value = [
            {"hadm_id": "100001", "chartdate": "2020-01-15", "text": "chest pain"}
        ]
        
        # Test analysis
        result = self.pipeline.process_patient("12345")
        
        # Assertions
        self.assertIn("mi_diagnosis", result)
        self.assertIn("onset_date", result)
        self.assertIn("evidence_summary", result)
    
    def test_patient_not_found(self):
        """Test handling of non-existent patient."""
        self.mock_patient_loader.get_patient_visit_history.return_value = []
        
        with self.assertRaises(ValueError):
            self.pipeline.process_patient("99999")

if __name__ == '__main__':
    unittest.main()
```

### 2. Integration Testing

```python
import pytest
from lighten_ml.pipeline import PatientLevelClinicalPipeline

@pytest.fixture
def pipeline():
    """Create pipeline for testing."""
    return create_mi_analysis_pipeline(
        "test_data/discharge_notes.csv",
        "test_data/labevents.csv",
        "test_data/d_labitems.csv"
    )

def test_end_to_end_analysis(pipeline):
    """Test complete end-to-end analysis."""
    result = pipeline.process_patient("test_patient_001")
    
    assert result["mi_diagnosis"] in ["POSITIVE", "NEGATIVE"]
    assert "onset_date" in result
    assert "evidence_summary" in result

def test_batch_processing(pipeline):
    """Test batch processing functionality."""
    patient_ids = ["test_patient_001", "test_patient_002"]
    results = []
    
    for patient_id in patient_ids:
        result = pipeline.process_patient(patient_id)
        results.append(result)
    
    assert len(results) == len(patient_ids)
    assert all("mi_diagnosis" in result for result in results)
```

## Monitoring and Observability

### 1. Metrics Collection

```python
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
analysis_requests_total = Counter('mi_analysis_requests_total', 'Total analysis requests', ['status'])
analysis_duration_seconds = Histogram('mi_analysis_duration_seconds', 'Analysis duration')
active_analyses = Gauge('mi_analysis_active', 'Active analyses')

class MonitoredMIAnalysisService:
    """MI analysis service with monitoring."""
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def analyze_patient(self, patient_id):
        """Analyze patient with monitoring."""
        active_analyses.inc()
        start_time = time.time()
        
        try:
            result = self.pipeline.process_patient(patient_id)
            analysis_requests_total.labels(status='success').inc()
            return result
            
        except Exception as e:
            analysis_requests_total.labels(status='error').inc()
            raise
            
        finally:
            duration = time.time() - start_time
            analysis_duration_seconds.observe(duration)
            active_analyses.dec()

# Start metrics server
start_http_server(8001)
```

### 2. Logging Integration

```python
import logging
import structlog
from pythonjsonlogger import jsonlogger

# Configure structured logging
def setup_logging():
    """Set up structured logging for production."""
    
    # JSON formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler('mi_analysis.log')
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Usage
setup_logging()
logger = structlog.get_logger()

def analyze_with_logging(pipeline, patient_id):
    """Analyze patient with structured logging."""
    logger.info("Starting MI analysis", patient_id=patient_id)
    
    try:
        result = pipeline.process_patient(patient_id)
        logger.info(
            "MI analysis completed",
            patient_id=patient_id,
            diagnosis=result["mi_diagnosis"],
            onset_date=result["onset_date"]
        )
        return result
        
    except Exception as e:
        logger.error(
            "MI analysis failed",
            patient_id=patient_id,
            error=str(e),
            exc_info=True
        )
        raise
```

## Deployment Considerations

### 1. Security

- **Data Encryption**: Encrypt sensitive patient data at rest and in transit
- **Access Control**: Implement proper authentication and authorization
- **Audit Logging**: Log all access and analysis requests
- **HIPAA Compliance**: Ensure compliance with healthcare data regulations

### 2. Scalability

- **Horizontal Scaling**: Use load balancers and multiple service instances
- **Database Optimization**: Implement proper indexing and query optimization
- **Caching**: Use Redis or similar for frequently accessed data
- **Async Processing**: Implement message queues for batch processing

### 3. Reliability

- **Error Handling**: Implement comprehensive error handling and recovery
- **Circuit Breakers**: Prevent cascade failures
- **Health Checks**: Implement proper health monitoring
- **Backup and Recovery**: Regular data backups and disaster recovery plans

This integration guide provides a comprehensive foundation for implementing the patient-level MI diagnosis system in various environments and use cases.
