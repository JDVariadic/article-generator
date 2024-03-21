
FROM python:3.12.2

#COPY . .

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

Run useradd -m -u 1000 useradd

USER user 
ENV Home=home/user \
    Path=/home/user/.local/bin:$Path

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]