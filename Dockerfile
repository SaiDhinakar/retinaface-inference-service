FROM python:3.10-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock ./

# Sync dependencies
RUN uv sync

# Create volumes for db and logs
VOLUME ["/app/db", "/app/logs"]

# Expose port if needed (assuming default for inference service)
EXPOSE 8000

# Command to run the service
CMD ["uv", "run", "fastapi", "run", "api/main.py"]