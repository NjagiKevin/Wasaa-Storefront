# Use official Airflow image as base
FROM apache/airflow:2.7.1-python3.11

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV PYTHONPATH="${AIRFLOW_HOME}/dags:${PYTHONPATH}"

# Copy DAGs and services into container
COPY ./dags ${AIRFLOW_HOME}/dags
COPY ./app ${AIRFLOW_HOME}/app
COPY ./requirements.txt ${AIRFLOW_HOME}/requirements.txt

# Install Python dependencies
USER root
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r ${AIRFLOW_HOME}/requirements.txt

# Switch back to airflow user
USER airflow
