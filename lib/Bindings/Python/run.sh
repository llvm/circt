#!/bin/bash

echo "================================================================================"
echo "FILE: ./__init__.py"
echo "================================================================================"
cat ./__init__.py
echo -e "\n\n"

echo "================================================================================"
echo "FILE: ./dialects/hw.py"
echo "================================================================================"
cat ./dialects/hw.py
echo -e "\n\n"

echo "================================================================================"
echo "FILE: ./dialects/__init__.py (if exists)"
echo "================================================================================"
if [ -f ./dialects/__init__.py ]; then
    cat ./dialects/__init__.py
else
    echo "File does not exist"
fi
echo -e "\n\n"

echo "================================================================================"
echo "FILE: ./CMakeLists.txt"
echo "================================================================================"
cat ./CMakeLists.txt
echo -e "\n\n"

echo "================================================================================"
echo "Checking what's in circt._mlir_libs._mlir.dialects"
echo "================================================================================"
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from circt._mlir_libs._mlir import dialects
    print('Available dialects:', dir(dialects))
except Exception as e:
    print(f'Error: {e}')
"
