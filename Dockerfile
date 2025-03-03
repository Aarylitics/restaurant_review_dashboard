FROM python:3.10-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y libstdc++6 chromium chromium-driver

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_reviews_menu.py"]
