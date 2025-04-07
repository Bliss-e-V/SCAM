FROM ollama/ollama:0.6.3 AS base

RUN apt-get update

# Install dependencies for python3.10
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    wget \
    libbz2-dev

# Download Python 3.10.14 source code
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -xvf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall

# Verify the installation
RUN python3.10 --version

# Clean up
RUN rm -rf Python-3.10.14.tgz Python-3.10.14 && \
    apt-get remove -y wget build-essential && \
    apt-get autoremove -y && \
    apt-get clean

RUN apt-get install -y python3-pip
RUN apt-get install -y htop tmux

FROM base
COPY requirements.txt .
RUN python3.10 -m pip install -U pip
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

COPY . .
