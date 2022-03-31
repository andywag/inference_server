
if [[ -z "${INFERENCE_BUILD}" ]]; then
    echo "Need to set INFERENCE_BUILD environment variable"
    #exit 1
fi
pushd $INFERENCE_BUILD
mkdir triton
pushd triton

#pip3 install --verbose --force --upgrade cmake
# Hack to put cmake to the correct version
#alias cmake=/localdata/andyw/projects/venv/bin/cmake
sudo apt-get install libre2-dev
sudo apt-get install -y rapidjson-dev
sudo apt-get install libarchive-dev
sudo apt-get install libssl-dev
sudo apt-get install build-essential libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev
sudo apt-get install libnuma-dev

#cd triton
#git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
#pushd mlperf_inference/loadgen
#CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
#pip install --force-reinstall dist/*.whl
#popd


git clone https://github.com/triton-inference-server/server.git 
pushd server
python3 build.py --build-dir mybuild --no-container-build --endpoint=grpc --enable-logging --enable-stats --install-dir install
cp -r ./mybuild/tritonserver/build/triton-server/mybuild/tritonserver/install/ .
popd

git clone https://github.com/triton-inference-server/python_backend.git
cd python_backend
mkdir build
cd build
cmake -DTRITON_ENABLE_GPU=OFF -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make install 
mkdir ../python
cp libtriton_python.so triton_python_backend_stub ../python/
chmod +x ../python/triton_python_backend_stub
popd