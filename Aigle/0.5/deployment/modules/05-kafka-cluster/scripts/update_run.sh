#!/bin/bash
# Kafka KRaft initialization script
# /modules/03-kafka-cluster/scripts/update_run.sh

# Set cluster ID
CLUSTER_ID="${CLUSTER_ID:-MkU3OEVBNTcwNTJENDM2Qk}"

# Format storage if needed
if [ ! -f /var/lib/kafka/data/meta.properties ]; then
    echo "Formatting Kafka storage with cluster ID: ${CLUSTER_ID}"
    kafka-storage format -t "${CLUSTER_ID}" -c /etc/kafka/kraft/server.properties --ignore-formatted
fi

# Start Kafka
exec /etc/confluent/docker/run
