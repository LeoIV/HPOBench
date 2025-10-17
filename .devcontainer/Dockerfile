# syntax=docker/dockerfile:1.4

# Base image (you can use Ubuntu, Debian, or another distro)
FROM ubuntu:latest

# Ensure non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive
ENV CACHEBUSTER=3
# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
        curl \
        git \
        python3 \
        python3-pip \
        make \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-openssl && rm -rf /var/lib/apt/lists/*
# Set up pyenv environment variables
ENV PYENV_ROOT="/home/ubuntu/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"

# Install pyenv
RUN curl https://pyenv.run | bash
# Initialize pyenv (make shims available)
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

# Install Python 3.11 using pyenv
ARG PYTHON_VERSION=3.11.13
RUN pyenv install $PYTHON_VERSION && pyenv global $PYTHON_VERSION
RUN python --version

# Install Poetry, set poetry home to local user directory
ENV POETRY_HOME="/home/ubuntu/.local/share/pypoetry"
ENV POETRY_VENV="/home/ubuntu/.cache/pypoetry/virtualenvs"
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"
RUN poetry config virtualenvs.create false

# Clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*

# make sure poetry and pyenv are also available in the running container


# Optional: set up your working directory
WORKDIR /workspace

# Copy requirements if you have them
# COPY requirements.txt .
# RUN pip install -r requirements.txt

# Default shell
CMD ["/bin/bash"]
