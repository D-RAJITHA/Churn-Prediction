FROM ubuntu:18.04

RUN apt-get update --fix-missing -y && apt-get install -y build-essential cmake \
libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
python3 python3-pip python3-dev && pip3 install --upgrade pip

# Set the locale
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
ADD . /app
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python3"]
CMD ["churn.py"]

