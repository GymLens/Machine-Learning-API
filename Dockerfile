FROM python:3.9-slim

WORKDIR /app

# Create upload directory
RUN mkdir -p /app/uploaded_images

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make sure upload directory has proper permissions
RUN chmod 777 /app/uploaded_images

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app