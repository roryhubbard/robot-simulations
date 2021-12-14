FROM robotlocomotion/drake:latest

WORKDIR /src

#RUN apt-get update && apt-get install -y \
#  python-pip \ # for pip install
#  libzmq3-dev \ # maybe needed for meschat
#  libgl1-mesa-glx \
#  libgl1-mesa-dri && \
#  rm -rf /var/lib/apt/lists/*

COPY . .

#RUN pip install -r requirements.txt

CMD ["python3", "main.py"]

