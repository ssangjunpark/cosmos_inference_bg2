FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
WORKDIR /opt/cosmos_gateway
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_PYTHON_REFRESH=quiet
COPY requirements.lock ./
COPY third_party/zenoh_ros2_sdk ./third_party/zenoh_ros2_sdk
RUN python -m pip install --no-cache-dir -r requirements.lock ./third_party/zenoh_ros2_sdk && python -m pip check
COPY scripts ./scripts
COPY messages ./messages
ENTRYPOINT ["python3", "-B", "/opt/cosmos_gateway/scripts/gateway.py"]
