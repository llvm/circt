// RUN: circt-opt -lower-handshake-to-firrtl -split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.module @handshake_memory_3ins_3outs_ui8
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
// CHECK: %[[MEM_LOAD:.+]], %[[MEM_STORE:.+]] = firrtl.mem Old {depth = 10 : i64, name = "mem0", portNames = ["load0", "store0"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: flip<uint<4>>, en: flip<uint<1>>, clk: flip<clock>, data: uint<8>>, !firrtl.flip<bundle<addr: uint<4>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>>

// Connect the store clock.
// CHECK: %[[MEM_LOAD_CLK:.+]] = firrtl.subfield %[[MEM_LOAD]]("clk") : {{.*}} -> !firrtl.flip<clock>
// CHECK: firrtl.connect %[[MEM_LOAD_CLK]], %clock

// Connect the load address, truncating if necessary.
// CHECK: %[[MEM_LOAD_ADDR:.+]] = firrtl.subfield %[[MEM_LOAD]]("addr") : {{.*}} -> !firrtl.flip<uint<4>>
// CHECK: %[[LD_ADDR_DATA_TAIL:.+]] = firrtl.tail %[[LD_ADDR_DATA]], 60 : (!firrtl.uint<64>) -> !firrtl.uint<4>
// CHECK: firrtl.connect %[[MEM_LOAD_ADDR]], %[[LD_ADDR_DATA_TAIL]]

// Connect the load data.
// CHECK: %[[MEM_LOAD_DATA:.+]] = firrtl.subfield %[[MEM_LOAD]]("data") : {{.*}} -> !firrtl.uint<8>
// CHECK: firrtl.connect %[[LD_DATA_DATA]], %[[MEM_LOAD_DATA]]

// Connect the load address valid to the load enable.
// CHECK: %[[MEM_LOAD_EN:.+]] = firrtl.subfield %[[MEM_LOAD]]("en") : {{.*}} -> !firrtl.flip<uint<1>>
// CHECK: firrtl.connect %[[MEM_LOAD_EN]], %[[LD_ADDR_VALID]]

// Create control-only fork for the load address valid and ready signal to the
// data and control signals. This re-uses the logic tested in test_fork.mlir, so
// the checks here are just at the module boundary.
// CHECK-DAG: firrtl.{{.+}} %[[LD_ADDR_VALID]]
// CHECK-DAG: firrtl.connect %[[LD_ADDR_READY]]
// CHECK-DAG: firrtl.connect %[[LD_DATA_VALID]]
// CHECK-DAG: firrtl.{{.+}} %[[LD_DATA_READY]]
// CHECK-DAG: firrtl.connect %[[LD_CONTROL_VALID]]
// CHECK-DAG: firrtl.{{.+}} %[[LD_CONTROL_READY]]

// Connect the store clock.
// CHECK: %[[MEM_STORE_CLK:.+]] = firrtl.subfield %[[MEM_STORE]]("clk") : {{.*}} -> !firrtl.flip<clock>
// CHECK: firrtl.connect %[[MEM_STORE_CLK]], %clock

// Connect the store address, truncating if necessary.
// CHECK: %[[MEM_STORE_ADDR:.+]] = firrtl.subfield %[[MEM_STORE]]("addr") : {{.*}} -> !firrtl.flip<uint<4>>
// CHECK: %[[ST_ADDR_DATA_TAIL:.+]] = firrtl.tail %[[ST_ADDR_DATA]], 60 : (!firrtl.uint<64>) -> !firrtl.uint<4>
// CHECK: firrtl.connect %[[MEM_STORE_ADDR]], %[[ST_ADDR_DATA_TAIL]]

// Connect the store data.
// CHECK: %[[MEM_STORE_DATA:.+]] = firrtl.subfield %[[MEM_STORE]]("data") : {{.*}} -> !firrtl.flip<uint<8>>
// CHECK: firrtl.connect %[[MEM_STORE_DATA]], %[[ST_DATA_DATA]]

// Create the write valid buffer.
// CHECK: %[[WRITE_VALID_BUFFER:.+]] = firrtl.regreset {{.+}} -> !firrtl.uint<1>

// Connect the write valid buffer to the store control valid.
// CHECK: firrtl.connect %[[ST_CONTROL_VALID]], %[[WRITE_VALID_BUFFER]]

// Create the store completed signal.
// CHECK: %[[STORE_COMPLETED:storeCompleted]] = firrtl.wire : !firrtl.uint<1>
// CHECK: %[[STORE_COMPLETED_GATE:.+]] = firrtl.and %[[ST_CONTROL_READY]], %[[WRITE_VALID_BUFFER]]
// CHECK: firrtl.connect %[[STORE_COMPLETED]], %[[STORE_COMPLETED_GATE]]

// Create the logic to drive the write valid buffer or keep its output.
// CHECK: %[[NOT_WRITE_VALID_BUFFER:.+]] = firrtl.not %[[WRITE_VALID_BUFFER]]
// CHECK: %[[EMPTY_OR_COMPLETE:.+]] = firrtl.or %[[NOT_WRITE_VALID_BUFFER]], %[[STORE_COMPLETED]]

// Connect the store data and address ports.
// CHECK: firrtl.connect %[[ST_ADDR_READY]], %[[EMPTY_OR_COMPLETE]]
// CHECK: firrtl.connect %[[ST_DATA_READY]], %[[EMPTY_OR_COMPLETE]]

// CHECK: %[[WRITE_VALID:writeValid]] = firrtl.wire : !firrtl.uint<1>
// CHECK: %[[WRITE_VALID_GATE:.+]] = firrtl.and %[[ST_DATA_VALID]], %[[ST_ADDR_VALID]]
// CHECK: firrtl.connect %[[WRITE_VALID]], %[[WRITE_VALID_GATE]]
// CHECK: %[[WRITE_VALID_BUFFER_MUX:.+]] = firrtl.mux(%[[EMPTY_OR_COMPLETE]], %[[WRITE_VALID]], %[[WRITE_VALID_BUFFER]])
// CHECK: firrtl.connect %[[WRITE_VALID_BUFFER]], %[[WRITE_VALID_BUFFER_MUX]]

// Connect the write valid signal to the memory enable
// CHECK: %[[MEM_STORE_EN:.+]] = firrtl.subfield %[[MEM_STORE]]("en") : {{.*}} -> !firrtl.flip<uint<1>>
// CHECK: firrtl.connect %[[MEM_STORE_EN]], %[[WRITE_VALID]]

// Connect the write valid signal to the memory mask.
// CHECK: %[[MEM_STORE_MASK:.+]] = firrtl.subfield %[[MEM_STORE]]("mask") : {{.*}} -> !firrtl.flip<uint<1>>
// CHECK: firrtl.connect %[[MEM_STORE_MASK]], %[[WRITE_VALID]]

// CHECK-LABEL: firrtl.module @main
handshake.func @main(%arg0: i8, %arg1: index, %arg2: index, ...) -> (i8, none, none) {
  // CHECK: %inst_arg0, %inst_arg1, %inst_arg2, %inst_arg3, %inst_arg4, %inst_arg5, %inst_clock, %inst_reset
  // CHECK: = firrtl.instance @handshake_memory_3ins_3outs_ui8
  %0:3 = "handshake.memory"(%arg0, %arg1, %arg2) {id = 0 : i32, ld_count = 1 : i32, lsq = false, st_count = 1 : i32, type = memref<10xi8>} : (i8, index, index) -> (i8, none, none)

  handshake.return %0#0, %0#1, %0#2: i8, none, none
}
