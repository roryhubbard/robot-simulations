FROM robotlocomotion/drake:latest

WORKDIR /src

#RUN apt-get update && apt-get install -y python-pip
#    libzmq3-dev && \
#    libgl1-mesa-glx \
#    libgl1-mesa-dri && \
#    rm -rf /var/lib/apt/lists/*

COPY . .

#RUN pip install -r requirements.txt

CMD ["python3", "main.py"]

