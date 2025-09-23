# Use official Airflow image as base
FROM apache/airflow:2.7.1-python3.11

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV PYTHONPATH="${AIRFLOW_HOME}/dags:${AIRFLOW_HOME}/app:${PYTHONPATH}"

# Requirements are copied; DAGs are bind-mounted at runtime to avoid duplication
COPY ./requirements.txt ${AIRFLOW_HOME}/requirements.txt

# Install Python dependencies
USER airflow
RUN pip install --no-cache-dir --user --upgrade pip \
    && pip install --no-cache-dir --user -r ${AIRFLOW_HOME}/requirements.txt
