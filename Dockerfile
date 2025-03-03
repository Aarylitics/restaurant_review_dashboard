FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

# 1. Update package lists and install ping
RUN apt-get update && apt-get install -y iputils-ping

# 2. Install core dependencies
RUN apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    libstdc++6

# 3. Install Chrome and related dependencies
RUN apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    wget \
    libgconf-2-4 \
    unzip \
    ca-certificates \
    libdbus-glib-1-2 \
    xvfb \
    libxcb-glx0 \
    libxshmfence1 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2

# 4. Download and install Chrome .deb with error handling, ping test and retry logic.
RUN ping -c 4 dl.google.com && \
    wget --tries=3 --waitretry=5 https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O chrome.deb || (echo "Download failed" && exit 1) && \
    dpkg -i chrome.deb && \
    rm chrome.deb

# 5. Fix broken dependencies
RUN apt-get -y --fix-broken install

# 6. Clean up apt cache
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 7. Install ChromeDriver for Chrome 133
RUN wget https://chromedriver.storage.googleapis.com/133.0.6943.142/chromedriver_linux64.zip && \
    unzip chromedriver_linux64.zip && \
    chmod +x chromedriver && \
    mv chromedriver /usr/local/bin/

# 8. Install GeckoDriver (Firefox)
RUN wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz && \
    tar -xzf geckodriver-v0.34.0-linux64.tar.gz && \
    chmod +x geckodriver && \
    mv geckodriver /usr/local/bin/

# 9. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_reviews_menu.py"]
