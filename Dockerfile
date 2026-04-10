FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY tightwad/ tightwad/
RUN pip install --no-cache-dir .
EXPOSE 8088
ENTRYPOINT ["tightwad", "proxy", "start"]
