# Stage 1: Build the wheel
FROM python:3.12-slim AS builder

WORKDIR /app

# Install git for setuptools_scm versioning
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Install build tools and build the wheel
RUN pip install --no-cache-dir build && \
    python -m build

# Stage 2: Runtime environment
FROM python:3.12-slim

WORKDIR /app

# Install runtime system dependencies
# libgomp1 is often needed for multi-threaded scientific libraries like numpy/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the wheel from the builder stage
COPY --from=builder /app/dist/*.whl .

# Install the wheel and dependencies
# This will install torch, numpy, h5py etc.
RUN pip install --no-cache-dir *.whl

# Clean up wheel file
RUN rm *.whl

# Set the default command to show help
ENTRYPOINT ["boost_corr"]
CMD ["--help"]
