From nvcr.io/nvidia/pytorch:19.04-py3

RUN apt-get -y update

RUN apt-get install -y python3-pip software-properties-common wget ffmpeg

RUN apt-get -y update

RUN mkdir -p /root/.torch/models

RUN mkdir -p /data/models

RUN wget -O /root/.torch/models/vgg16_bn-6c64b313.pth https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

RUN wget -O /root/.torch/models/resnet34-333f7ec4.pth https://download.pytorch.org/models/resnet34-333f7ec4.pth

ADD . /data/

#incase models/ColorizeArtistic_gen.pth is already available

COPY /models/ColorizeArtistic_gen.pth /data/models/ColorizeArtistic_gen.pth

#RUN wget -O /data/models/ColorizeArtistic_gen.pth https://www.dropbox.com/s/zkehq1uwahhbc2o/ColorizeArtistic_gen.pth?dl=0 

WORKDIR /data

RUN pip install -r requirements.txt

RUN pip install  Flask

RUN pip install Pillow

RUN pip install scikit-image

RUN pip install requests

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["app.py"]

