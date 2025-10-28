FROM prom/alertmanager:v0.28.1

# Copy the configuration and entrypoint script into the image
COPY ./alertmanager.yml /etc/alertmanager/alertmanager.yml
COPY ./entrypoint.sh /entrypoint.sh

# Copy the entrypoint script and set it as executable with octal permissions
COPY --chmod=0755 entrypoint.sh /entrypoint.sh

# Use the custom entrypoint
ENTRYPOINT ["/entrypoint.sh"]