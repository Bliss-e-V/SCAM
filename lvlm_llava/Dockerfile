FROM registry.datexis.com/jwesterhoff/llava-custom-cuda118:latest

# Install packages used in evaluation (adjust as needed)
RUN pip install pandas tqdm datasets httpx

# Copy evaluation script
COPY main_lvlm_llava.py /app/main_lvlm_llava.py
COPY utils.py /app/utils.py

# Set path to locate LLaVA code - initialize PYTHONPATH if it doesn't exist
ENV PYTHONPATH="/app:/src:/llava:/LLaVA"

WORKDIR /app
# No CMD - will be provided by kubernetes job configuration
