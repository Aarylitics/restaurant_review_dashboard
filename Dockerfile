FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential python3-dev libstdc++6 chromium chromium-driver

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_reviews_menu.py"]
