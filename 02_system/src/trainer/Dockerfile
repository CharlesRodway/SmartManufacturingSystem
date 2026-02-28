FROM python:3.11-slim
WORKDIR /app
COPY trainer.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY trainer.py .
CMD ["python", "trainer.py"]