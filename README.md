# 🚀 MLOps Project - Complete Pipeline

A professional MLOps pipeline implementing FastAPI, MLflow, Prometheus, and comprehensive testing.

## 📋 Project Overview
- **Student**: Rayen Kaddechi
- **Class**: 4DS10  
- **Status**: Production-Ready MLOps Pipeline

## 🛠️ Features
- ✅ FastAPI with Swagger Documentation
- ✅ MLflow Experiment Tracking
- ✅ Model Persistence (.pkl files)
- ✅ Prometheus Metrics Monitoring
- ✅ Comprehensive Testing (12/12 tests passing)
- ✅ CI/CD Pipeline (GitHub Actions + Jenkins)
- ✅ Docker Containerization

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone and setup
git clone <your-repo>
cd Kaddechi-rayen-4DS10-ml_project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Start API server
uvicorn src.api:app --reload --host 0.0.0.0 --port 8001

# 5. Start MLflow UI (new terminal)
mlflow ui --host 0.0.0.0 --port 5000
