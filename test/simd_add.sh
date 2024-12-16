echo "[1,2,3,6]+[2,3,8,5]" >> input.txt
../build/cpp-peglib/mlir/mlir-calc
rm -rf input.txt