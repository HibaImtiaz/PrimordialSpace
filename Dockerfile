FROM python:3.8.8

WORKDIR /opt/ml_api

ENV FLSK_APP flask_app.py

#install requirements
COPY ./Flask_API /opt/ml_api/
RUN pip install --upgrade pip
RUN pip install -r /opt/ml_api/Requirements.txt

EXPOSE 5000

CMD ["python","./flask_app.py"]
