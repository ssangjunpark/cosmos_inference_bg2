ARG ROBOTIS_IMAGE=robotis/ai-worker:2.2.8
FROM ${ROBOTIS_IMAGE}
# Add only this companion. No driver changes or package installation.
COPY cosmos_ffw /opt/cosmos_ffw/cosmos_ffw
COPY run.sh /opt/cosmos_ffw/run.sh
ENV PYTHONPATH=/opt/cosmos_ffw PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["/bin/bash", "/opt/cosmos_ffw/run.sh"]
