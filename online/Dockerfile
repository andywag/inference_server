# set base image (host OS)
FROM python:3.9

#RUN apt install libgl1-mesa-glx -y
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6 -y
# set the working directory in the container
WORKDIR /online

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY model_proto/ ./model_proto
COPY public_api/ ./public_api
COPY release/ ./release

WORKDIR /online/release
# command to run on container start
CMD [ "python", "./fastapi_runner.py" ]