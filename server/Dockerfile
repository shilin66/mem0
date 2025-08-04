FROM python:3.12-slim

WORKDIR /app

# Copy mem0 source code first
COPY ../mem0 ./mem0
COPY ../pyproject.toml ./pyproject.toml
COPY ../poetry.lock ./poetry.lock

# Install mem0 from source
RUN pip install --no-cache-dir -e .

# Copy server requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY . .

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
