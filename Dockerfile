FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# Install Chromium and dependencies
RUN apt-get update && apt-get install -y \
    chromium chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set up environment variables to use Chromium
ENV CHROMIUM_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_BIN=/usr/bin/chromedriver

# Copy your project files into the container
COPY . .

# EXPOSE 8501
#CMD ["streamlit", "run", "streamlit_reviews_menu.py"]

# Ensure Streamlit listens on the correct port and IP address
CMD ["streamlit", "run", "streamlit_reviews_menu.py", "--server.port=$PORT", "--server.address=0.0.0.0"]


