#!/bin/sh
set -e

# Substitute placeholders in filer.toml 
sed \
  -e "s|\${MYSQL_HOST}|${MYSQL_HOST}|g" \
  -e "s|\${MYSQL_PORT}|${MYSQL_PORT}|g" \
  -e "s|\${MYSQL_USER}|${MYSQL_USER}|g" \
  -e "s|\${MYSQL_PASSWORD}|${MYSQL_PASSWORD}|g" \
  -e "s|\${MYSQL_DATABASE}|${MYSQL_DATABASE}|g" \
  -e "s|\${S3_BUCKET}|${S3_BUCKET}|g" \
  /etc/seaweedfs/filer_generated.toml > /etc/seaweedfs/filer.toml

# Run filer
exec weed filer -master=seaweedfs-master1:9333,seaweedfs-master2:9334,seaweedfs-master3:9335 \
    -ip=seaweedfs-filer -ip.bind=0.0.0.0 \
    -port=8888 -port.grpc=18888 \
    -s3.allowEmptyFolder=false -encryptVolumeData \
    -metricsPort=1241
