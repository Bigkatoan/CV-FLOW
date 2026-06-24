FROM python:3.11-slim

# Install C++ build tools for on-the-fly compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake make \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
