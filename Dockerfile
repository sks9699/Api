FROM python:3-alpine3.10
WORKDIR . /app
RUN pip install -r requirements.txt
EXPOSE 3000
CMD python ./index.py