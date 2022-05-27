// REQUIRES: ieee-sim
// UNSUPPORTED: ieee-sim-iverilog
// RUN: circt-opt %s \
// RUN:   --canonicalize='top-down=true region-simplify=true' \
// RUN:   --handshake-materialize-forks-sinks --canonicalize \
// RUN:   --handshake-insert-buffers=strategy=all --lower-handshake-to-firrtl | \
// RUN: firtool --format=mlir --verilog > %buffer-export.sv
// RUN: circt-rtl-sim.py %buffer-export.sv %S/driver.sv --sim %ieee-sim --no-default-driver --top driver | FileCheck %s
// CHECK: Result={{.*}}10

module {
  handshake.func @top(%arg0: none) -> (i32, none) attributes {argNames = ["inCtrl"], resNames = ["out0", "outCtrl"]} {
    %d = buffer [2] seq %dTrue {initValues = [20, 10]} : i32
    %ctrl = merge %arg0, %cTrue : none

    %6 = constant %ctrl {value = 10 : i32} : i32
    %cond = arith.cmpi ne, %d, %6 : i32

    %dTrue, %dFalse = cond_br %cond, %d : i32
    %cTrue, %cFalse = cond_br %cond, %ctrl : none

    return %dFalse, %cFalse : i32, none
  }
}
