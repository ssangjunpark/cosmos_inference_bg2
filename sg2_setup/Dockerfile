# Build with the pinned zenoh_ros2_sdk checkout as context.
FROM cosmos-framework:robotescape-b98934e
COPY . /opt/sg2-zenoh-sdk
RUN uv pip install --python /workspace/.venv/bin/python /opt/sg2-zenoh-sdk \
    eclipse-zenoh==1.10.2 rosbags==0.11.9 Pillow==12.2.0
