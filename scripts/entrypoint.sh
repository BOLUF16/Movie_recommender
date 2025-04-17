#!/bin/bash
set -e

if [ -e "/opt/airflow/requirements.txt" ]; then
  $(command -v pip) install --upgrade pip
  $(command -v pip) install --user -r /opt/airflow/requirements.txt
fi

# Initialize Airflow database if it doesn't exist
if [ ! -f "/opt/airflow/airflow.db" ]; then
  airflow db init && \
  airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com \
    --password admin
fi

# Upgrade the database
$(command -v airflow) db upgrade

# Start the webserver
exec airflow webserver