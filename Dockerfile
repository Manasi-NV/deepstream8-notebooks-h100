# DeepStream 8.0 with Python Bindings + Jupyter Lab
# Pre-configured with object detection notebooks
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python3-gi python3-gst-1.0 \
    python-gi-dev git cmake g++ build-essential \
    libglib2.0-dev libgstreamer1.0-dev libgirepository1.0-dev libcairo2-dev \
    pybind11-dev && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /opt/nvidia/deepstream/deepstream-8.0/sources && \
    git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
    cd deepstream_python_apps && \
    git submodule update --init && \
    cd bindings && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    cp pyds.so /usr/local/lib/python3.12/dist-packages/

COPY notebooks/ /app/notebooks/

ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream-8.0/sources/deepstream_python_apps/apps"

# Expose Jupyter Lab port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks", "--NotebookApp.token=''", "--NotebookApp.password=''"]
