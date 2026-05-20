#!/bin/sh
set -e

# Substitute placeholders in filer.toml 
sed \
  -e "s|\${POSTGRES_HOST}|${POSTGRES_HOST}|g" \
  -e "s|\${POSTGRES_PORT}|${POSTGRES_PORT}|g" \
  -e "s|\${POSTGRES_USER}|${POSTGRES_USER}|g" \
  -e "s|\${POSTGRES_PASSWORD}|${POSTGRES_PASSWORD}|g" \
  -e "s|\${POSTGRES_DB}|${POSTGRES_DB}|g" \
  /etc/seaweedfs/filer_generated.toml > /etc/seaweedfs/filer.toml

# Run filer
exec weed filer -master=seaweedfs-master1:9333,seaweedfs-master2:9334,seaweedfs-master3:9335 \
    -ip=seaweedfs-filer -ip.bind=0.0.0.0 \
    -port=8888 -port.grpc=18888 \
    -s3.allowEmptyFolder=false -encryptVolumeData \
    -metricsPort=1241
