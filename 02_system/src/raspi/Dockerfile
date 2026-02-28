FROM python:3.11-slim
WORKDIR /app
COPY raspi.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY raspi.py .
CMD ["python", "raspi.py"]
