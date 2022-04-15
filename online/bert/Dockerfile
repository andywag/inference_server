FROM python:3.6

mkdir /home/triton
RUN git clone https://github.com/triton-inference-server/server.git /home/triton
cd triton_server
RUN python3 build.py --build-dir mybuild --no-container-build --endpoint=grpc --enable-logging --enable-stats --cmake-dir `pwd`/build

WORKDIR /home/triton
RUN git clone https://github.com/triton-inference-server/python_backend.git
cd python_backend
mkdir build
cd build
RUN cmake -DTRITON_ENABLE_GPU=OFF -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
RUN make install 

