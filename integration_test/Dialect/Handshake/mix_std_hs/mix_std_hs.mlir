// REQUIRES: iverilog,cocotb

// This test is executed with all different buffering strategies

// RUN: hlstool %s --dynamic-firrtl --buffering-strategy=all --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: hlstool %S/kernel.mlir --dynamic-firrtl --buffering-strategy=all --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=mix_std_hs --pythonFolder=%S %T/*.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --buffering-strategy=allFIFO --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: hlstool %S/kernel.mlir --dynamic-firrtl --buffering-strategy=allFIFO --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=mix_std_hs --pythonFolder=%S %T/*.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --buffering-strategy=cycles --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: hlstool %S/kernel.mlir --dynamic-firrtl --buffering-strategy=cycles --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=mix_std_hs --pythonFolder=%S %T/*.sv 2>&1 | FileCheck %s

// RUN: hlstool %s --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=locking --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: hlstool %S/kernel.mlir --dynamic-firrtl --buffering-strategy=all --dynamic-parallelism=locking --ir | firtool -format=mlir -o %T --split-verilog --lowering-options=disallowLocalVariables && \
// RUN: %PYTHON% %S/../cocotb_driver.py --objdir=%T --topLevel=top --pythonModule=mix_std_hs --pythonFolder=%S %T/*.sv 2>&1 | FileCheck %s

// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

!tupleType = tuple<i64,i64,i64,i64,i64,i64,i64,i64>

module {
  handshake.func @compute(i64, none) -> (i64, none)
  handshake.func @top(%arg0: !tupleType, %arg1: none, ...) -> (!tupleType, none) attributes {argNames = ["in0", "inCtrl"], resNames = ["out0", "outCtrl"]} {
    %vals:8 = handshake.unpack %arg0 : !tupleType
    %ctrls:8 = fork [8] %arg1 : none

    %res0, %c0 = handshake.instance @compute(%vals#0, %ctrls#0) : (i64, none) -> (i64, none)
    %res1, %c1 = handshake.instance @compute(%vals#1, %ctrls#1) : (i64, none) -> (i64, none)
    %res2, %c2 = handshake.instance @compute(%vals#2, %ctrls#2) : (i64, none) -> (i64, none)
    %res3, %c3 = handshake.instance @compute(%vals#3, %ctrls#3) : (i64, none) -> (i64, none)
    %res4, %c4 = handshake.instance @compute(%vals#4, %ctrls#4) : (i64, none) -> (i64, none)
    %res5, %c5 = handshake.instance @compute(%vals#5, %ctrls#5) : (i64, none) -> (i64, none)
    %res6, %c6 = handshake.instance @compute(%vals#6, %ctrls#6) : (i64, none) -> (i64, none)
    %res7, %c7 = handshake.instance @compute(%vals#7, %ctrls#7) : (i64, none) -> (i64, none)

    %res = handshake.pack %res0, %res1, %res2, %res3, %res4, %res5, %res6, %res7 : !tupleType
    %outCtrl = join %c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7 : none, none, none, none, none, none, none, none
    return %res, %outCtrl : !tupleType, none
  }
}

