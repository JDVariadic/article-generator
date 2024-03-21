
FROM python:3.12.2


COPY . .


WORKDIR /


RUN pip install --no-cache-dir --upgrade -r /requirements.txt


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]