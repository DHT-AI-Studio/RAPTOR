# FROM bitnamilegacy/spark:3.5
FROM apache/spark:3.5.7

USER root

# Install cron
RUN apt-get update && apt-get install -y cron && rm -rf /var/lib/apt/lists/*

# Copy GC script
COPY run-gc.sh /usr/local/bin/run-gc.sh
RUN chmod +x /usr/local/bin/run-gc.sh

# Copy entrypoint
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

CMD ["/usr/local/bin/entrypoint.sh"]
