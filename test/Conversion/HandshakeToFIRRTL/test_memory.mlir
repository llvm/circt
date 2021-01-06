// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_memory_3ins_3outs
// CHECK: %[[ST_DATA_VALID:.+]] = firrtl.subfield %arg0("valid")
// CHECK: %[[ST_DATA_READY:.+]] = firrtl.subfield %arg0("ready")
// CHECK: %[[ST_DATA_DATA:.+]] = firrtl.subfield %arg0("data")
// CHECK: %[[ST_ADDR_VALID:.+]] = firrtl.subfield %arg1("valid")
// CHECK: %[[ST_ADDR_READY:.+]] = firrtl.subfield %arg1("ready")
// CHECK: %[[ST_ADDR_DATA:.+]] = firrtl.subfield %arg1("data")
// CHECK: %[[LD_ADDR_VALID:.+]] = firrtl.subfield %arg2("valid")
// CHECK: %[[LD_ADDR_READY:.+]] = firrtl.subfield %arg2("ready")
// CHECK: %[[LD_ADDR_DATA:.+]] = firrtl.subfield %arg2("data")
// CHECK: %[[LD_DATA_VALID:.+]] = firrtl.subfield %arg3("valid")
// CHECK: %[[LD_DATA_READY:.+]] = firrtl.subfield %arg3("ready")
// CHECK: %[[LD_DATA_DATA:.+]] = firrtl.subfield %arg3("data")
// CHECK: %[[ST_CONTROL_VALID:.+]] = firrtl.subfield %arg4("valid")
// CHECK: %[[ST_CONTROL_READY:.+]] = firrtl.subfield %arg4("ready")
// CHECK: %[[LD_CONTROL_VALID:.+]] = firrtl.subfield %arg5("valid")
// CHECK: %[[LD_CONTROL_READY:.+]] = firrtl.subfield %arg5("ready")

// Construct the memory.
// CHECK: %[[MEM:.+]] = firrtl.mem "Old" {depth = 10 : i64, name = "mem0", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<load0: bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<8>>, store0: flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>>>

// Get the load0 port.
// CHECK: %[[MEM_LOAD:.+]] = firrtl.subfield %[[MEM]]("load0") : {{.*}} -> !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<8>>

// Connect the load address, truncating if necessary.
// CHECK: %[[MEM_LOAD_ADDR:.+]] = firrtl.subfield %[[MEM_LOAD]]("addr") : {{.*}} -> !firrtl.flip<uint<4>>
// CHECK: %[[LD_ADDR_DATA_TAIL:.+]] = firrtl.tail %[[LD_ADDR_DATA]], 60 : (!firrtl.uint<64>) -> !firrtl.uint<4>
// CHECK: firrtl.connect %[[MEM_LOAD_ADDR]], %[[LD_ADDR_DATA_TAIL]] : !firrtl.flip<uint<4>>, !firrtl.uint<4>

// Connect the load data.
// CHECK: %[[MEM_LOAD_DATA:.+]] = firrtl.subfield %[[MEM_LOAD]]("data") : {{.*}} -> !firrtl.uint<8>
// CHECK: firrtl.connect %[[LD_DATA_DATA]], %[[MEM_LOAD_DATA]] : !firrtl.flip<uint<8>>, !firrtl.uint<8>

// Create control-only fork for the load address valid and ready signal to the
// data and control signals. This re-uses the logic tested in test_fork.mlir, so
// the checks here are just at the module boundary.
// CHECK-DAG: firrtl.{{.+}} %[[LD_ADDR_VALID]]
// CHECK-DAG: firrtl.connect %[[LD_ADDR_READY]]
// CHECK-DAG: firrtl.connect %[[LD_DATA_VALID]]
// CHECK-DAG: firrtl.{{.+}} %[[LD_DATA_READY]]
// CHECK-DAG: firrtl.connect %[[LD_CONTROL_VALID]]
// CHECK-DAG: firrtl.{{.+}} %[[LD_CONTROL_READY]]

// CHECK-LABEL: firrtl.module @main
handshake.func @main(%arg0: i8, %arg1: index, %arg2: index, ...) -> (i8, none, none) {
  // CHECK: %0 = firrtl.instance @handshake_memory_3ins_3outs
  %0:3 = "handshake.memory"(%arg0, %arg1, %arg2) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xi8>} : (i8, index, index) -> (i8, none, none)

  handshake.return %0#0, %0#1, %0#2: i8, none, none
}
