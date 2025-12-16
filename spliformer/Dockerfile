FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git gcc g++ curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU (lighter than GPU for a sidecar)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Clone and install Spliformer
RUN git clone https://github.com/TJ-zhanglab/Spliformer.git && \
    cd Spliformer && \
    pip install --no-cache-dir -r requirements.txt && \
    python setup.py install

# Install the API wrapper deps
RUN pip install --no-cache-dir fastapi uvicorn pyfaidx pysam boto3

COPY app.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 5001
ENTRYPOINT ["./entrypoint.sh"]
