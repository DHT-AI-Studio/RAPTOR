FROM opensearchproject/opensearch:3.4.0

# Version must match the OpenSearch base image exactly (OpenSearch plugins
# are version-locked); see https://github.com/opensearch-project/opensearch-prometheus-exporter/releases
RUN ./bin/opensearch-plugin install --batch \
    https://github.com/opensearch-project/opensearch-prometheus-exporter/releases/download/3.4.0.0/prometheus-exporter-3.4.0.0.zip
