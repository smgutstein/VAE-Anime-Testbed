# Start from the official TensorFlow image with GPU support.
# This already includes TensorFlow 2.16.2 plus compatible CUDA/cuDNN runtime libraries.
FROM tensorflow/tensorflow:2.16.2-gpu

# Avoid interactive apt prompts during Docker builds.
# Disable pip's package cache to keep the image smaller.
# Prevent Python from writing __pycache__/.pyc files.
# Force Python logs/output to appear immediately in Docker logs.
# Reduce TensorFlow startup noise.
# Disable oneDNN CPU optimizations to reduce small numerical-order differences.
# Use a non-GUI Matplotlib backend suitable for containers.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    MPLBACKEND=Agg

# Use /workspace as the default project directory inside the container.
WORKDIR /workspace

# Install small OS-level dependencies.
# ca-certificates is needed for HTTPS access by pip, Python, git, etc.
# ffmpeg is useful for image/video generation or animation output.
# --no-install-recommends avoids unnecessary optional packages.
# Removing /var/lib/apt/lists keeps the final image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      ffmpeg \
      gosu \
      bash \
  && rm -rf /var/lib/apt/lists/*

# Update Python packaging tools before installing project dependencies.
# This helps avoid package build/install issues later.
RUN pip install --upgrade pip setuptools wheel

# Install Project Dependencies
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Copy then run the startup script into a directory on the container PATH.
# The entrypoint creates a container user matching the host UID/GID,
# then drops privileges before running the requested command.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Always do root-only setup, then switch to the normal user.
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Default command
CMD ["bash"]