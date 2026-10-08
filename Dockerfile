FROM python:3.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY model_export.zip Day3_Assignment_Submission.ipynb cloud_seed.py cloud_start.py /app/
RUN python cloud_seed.py
EXPOSE 10000
CMD ["python", "cloud_start.py"]
