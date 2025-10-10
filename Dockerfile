# Use the official TensorFlow CPU image (has Python + TensorFlow preinstalled)
FROM tensorflow/tensorflow:latest

# Set a working directory (Streamlit prefers non-root WORKDIR). All commands run from /app.
WORKDIR /app

# Set environment variables (explanations below)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Copy only dependency file first for better layer caching
COPY requirements2.txt ./ 

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip \
 && pip install --ignore-installed --no-cache-dir -r requirements2.txt


# Copy the rest of the code (app.py, src/, saved_models/)
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the Streamlit app; bind to 0.0.0.0 so it's accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
