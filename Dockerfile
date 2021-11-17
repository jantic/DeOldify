FROM nvcr.io/nvidia/pytorch:19.04-py3

RUN apt-get -y update && apt-get install -y \
	python3-pip \
	software-properties-common \
	wget \
	ca-certificates \
	libcurl4-openssl-dev \
	libssl-dev \
	ffmpeg

RUN apt-get clean
RUN update-ca-certificates -f

RUN mkdir -p /root/.torch/models

RUN mkdir -p /data/models

RUN wget -O /root/.torch/models/vgg16_bn-6c64b313.pth https://download.pytorch.org/models/vgg16_bn-6c64b313.pth

RUN wget -O /root/.torch/models/resnet34-333f7ec4.pth https://download.pytorch.org/models/resnet34-333f7ec4.pth



# if you want to avoid image building with downloading put your .pth file in root folder
COPY Dockerfile ColorizeArtistic_gen.* /data/models/
COPY Dockerfile ColorizeVideo_gen.* /data/models/

RUN pip install --upgrade pip \
	&& pip install versioneer==0.18 \
		tensorboardX==1.6 \
		Flask==1.1.1 \
		pillow==6.1 \
		numpy==1.16 \
		scikit-image==0.15.0 \
		requests==2.21.0 \
		ffmpeg-python==0.1.17 \
		youtube-dl>=2019.4.17 \
		jupyterlab==1.2.4 \
		opencv-python>=3.3.0.10 \
		fastai==1.0.51

ADD . /data/

WORKDIR /data

# force download of file if not provided by local cache
RUN [[ ! -f /data/models/ColorizeArtistic_gen.pth ]] && wget -O /data/models/ColorizeArtistic_gen.pth https://data.deepai.org/deoldify/ColorizeArtistic_gen.pth
RUN [[ ! -f /data/models/ColorizeVideo_gen.pth ]] && wget -O /data/models/ColorizeVideo_gen.pth https://data.deepai.org/deoldify/ColorizeVideo_gen.pth

COPY run_notebook.sh /usr/local/bin/run_notebook
COPY run_image_api.sh /usr/local/bin/run_image_api
COPY run_video_api.sh /usr/local/bin/run_video_api

RUN chmod +x /usr/local/bin/run_notebook
RUN chmod +x /usr/local/bin/run_image_api
RUN chmod +x /usr/local/bin/run_video_api

EXPOSE 8888
EXPOSE 5000

# run notebook
# ENTRYPOINT ["sh", "run_notebook"]

# run image api
# ENTRYPOINT ["sh", "run_image_api"]

# run image api
# ENTRYPOINT ["sh", "run_video_api"]
