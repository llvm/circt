source ../setenv.sh
rm -rf build
mkdir build
cd build
cmake -G Ninja .. -DMLIR_DIR=~/workspace/circt/llvm/build/lib/cmake/mlir -DLLVM_DIR=~/workspace/circt/llvm/build/lib/cmake/llvm -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-circt
cd -
