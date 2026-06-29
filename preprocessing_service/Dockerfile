FROM python:3.10-slim

WORKDIR /app

COPY preprocessing_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY preprocessing_service/app.py .
COPY preprocessing_service/bpm_processor.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
