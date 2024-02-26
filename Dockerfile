FROM python:3.10-slim as build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential gcc

WORKDIR /home/user

RUN python -m venv /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY ./app/requirements.txt ./app/requirements.txt
RUN python -m pip install --no-cache-dir -r ./app/requirements.txt

RUN useradd -ms /bin/bash user
USER user

COPY --chown=user:user ./app ./app
COPY --chown=user:user ./shared ./shared
COPY --chown=user:user fasttext-0.9.2-cp310-cp310-win_amd64.whl ./


ENV PYTHONPATH=/home/user
CMD ["/home/user/venv/bin/python", "./app/main.py"]
