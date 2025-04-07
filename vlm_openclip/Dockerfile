FROM python:3.10.14-bookworm

RUN apt-get update
RUN apt-get install -y htop tmux

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --timeout=10000 -r requirements.txt

COPY . .
