#!/bin/sh

file_path="/tmp/clusterID/clusterID"
interval=5  # wait interval in seconds

while [ ! -e "$file_path" ] || [ ! -s "$file_path" ]; do
  echo "Waiting for $file_path to be created..."
  sleep $interval
done

export CLUSTER_ID=$(cat "$file_path")

echo "Using CLUSTER_ID: $CLUSTER_ID"

# KRaft required step: Format the storage directory with a new cluster ID
echo "kafka-storage format --ignore-formatted -t $CLUSTER_ID -c /etc/kafka/kafka.properties"

exec /etc/confluent/docker/run

