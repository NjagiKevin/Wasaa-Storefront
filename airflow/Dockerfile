# Use official Airflow image as base
FROM apache/airflow:2.7.1-python3.11

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV PYTHONPATH="${AIRFLOW_HOME}/dags:${AIRFLOW_HOME}/app:${PYTHONPATH}"

# Switch to root to install system packages
USER root
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Airflow-specific requirements
COPY airflow/airflow_requirements.txt ${AIRFLOW_HOME}/requirements.txt

# Install Python dependencies
USER airflow
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r ${AIRFLOW_HOME}/requirements.txt
