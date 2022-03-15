
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
pushd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
popd


sudo apt-get install libre2-dev
sudo apt-get install -y rapidjson-dev
sudo apt-getinstall libarchive-dev
git clone https://github.com/triton-inference-server/server.git
pushd server
python3 build.py --build-dir mybuild --no-container-build --endpoint=grpc --enable-logging --enable-stats --cmake-dir `pwd`/build
cp -r ./mybuild/tritonserver/build/server/mybuild/tritonserver/install .
popd

git clone https://github.com/triton-inference-server/python_backend.git
cd python_backend
mkdir build
cd build
/snap/bin/cmake -DTRITON_ENABLE_GPU=OFF -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install 
mkdir ../python
cp libtriton_python.so triton_python_backend_stub ../python/
chmod +x ../python/triton_python_backend_stub