// REQUIRES: iverilog,cocotb

// Test 1: default lowering (input muxing)

// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(pipeline.scheduled(pipeline-explicit-regs), lower-pipeline-to-hw{outline-stages}), lower-seq-to-sv, sv-trace-iverilog, export-verilog)' \
// RUN:     -o %t.mlir > %t.sv

// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=stallTest \
// RUN:     --pythonModule=stallTest --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test 2: Clock-gate implementation

// RUN: circt-opt %s -pass-pipeline='builtin.module(hw.module(pipeline.scheduled(pipeline-explicit-regs), lower-pipeline-to-hw{outline-stages clock-gate-regs}), lower-seq-to-sv, sv-trace-iverilog, export-verilog)' \
// RUN:     -o %t_clockgated.mlir > %t.sv

// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=stallTest \
// RUN:     --pythonModule=stallTest --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s


// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @stallTest(%arg0 : i32, %arg1 : i32, %go : i1, %stall : i1, %clock : i1, %reset : i1) -> (out: i32, done : i1) {
  %out, %done = pipeline.scheduled(%arg0, %arg1) stall %stall clock %clock reset %reset go %go : (i32, i32) -> (i32) {
    ^bb0(%a0 : i32, %a1: i32, %s0_valid : i1):
      %add0 = comb.add %a0, %a1 : i32
      pipeline.stage ^bb1

    ^bb1(%s1_valid : i1):
      %add1 = comb.add %add0, %a0 : i32
      pipeline.stage ^bb2

    ^bb2(%s2_valid : i1):
      %add2 = comb.add %add1, %add0 : i32
      pipeline.return %add2 : i32
  }
  hw.output %out, %done : i32, i1
}
