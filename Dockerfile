FROM python:3.11.8

WORKDIR /app

COPY ./Agent /app/Agent
COPY ./tests /app/tests
COPY ./Agent/requirements.txt /app/Agent/requirements.txt

RUN pip install --no-cache-dir -r /app/Agent/requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8538

CMD ["python", "/app/tests/test_agent_rag.py"]