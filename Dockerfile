# Use the fully qualified Miniconda3 image from Docker Hub.
FROM docker.io/continuumio/miniconda3

# Install Python 3.11, tomopy, dxchange, and matplotlib from conda-forge.
RUN conda install -y -c conda-forge python=3.11 tomopy dxchange matplotlib && \
    conda clean -afy

# Install globus-compute-sdk via pip.
RUN pip install --no-cache-dir globus-compute-sdk

# Default command: launch an interactive Python shell.
CMD ["python3"]