FROM python:3.11-slim-buster


RUN mkdir /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN pip install "arkitekt-next[all]>=0.12.1"

COPY .arkitekt_next /app/.arkitekt_next
COPY app.py /app/app.py
