FROM python:3.11

WORKDIR /app

COPY *.py /app
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python weights_download.py

ENTRYPOINT ["python3", "app.py"]