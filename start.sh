#!/bin/bash

# Получение IP адреса spark-master
LOGS=$(docker logs spark-master 2>&1)
IP=$(echo "$LOGS" | grep -oP 'Starting Spark master at spark://\K[0-9.]+')

# Запуск Spark приложения
docker exec -it spark-master /opt/bitnami/spark/bin/spark-submit --master spark://$IP:7077 /scripts/single.py

