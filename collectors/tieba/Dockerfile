FROM python:3.12-slim
WORKDIR /app
COPY tieba_collector.py /app/tieba_collector.py
COPY run_daily.py /app/run_daily.py
RUN mkdir -p /app/data
ENTRYPOINT ["python3","/app/tieba_collector.py"]
