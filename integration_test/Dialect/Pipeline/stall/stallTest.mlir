// REQUIRES: iverilog,cocotb

// Test 1: default lowering (input muxing)

// RUN: circt-opt %s -pipeline-explicit-regs -lower-pipeline-to-hw -lower-seq-to-sv -sv-trace-iverilog -export-verilog \
// RUN:     -o %t.mlir > %t.sv

// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=stallTest \
// RUN:     --pythonModule=stallTest --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s

// Test 2: Clock-gate implementation

// RUN: circt-opt %s -pipeline-explicit-regs -lower-pipeline-to-hw="clock-gate-regs" -lower-seq-to-sv -sv-trace-iverilog -export-verilog \
// RUN:     -o %t_clockgated.mlir > %t.sv

// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=stallTest \
// RUN:     --pythonModule=stallTest --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s


// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @stallTest(in %arg0 : i32, in %arg1 : i32, in %go : i1, in %stall : i1, in %clock : !seq.clock, in %reset : i1, out out: i32, out done : i1) {
  %out, %done = pipeline.scheduled(%a0 : i32 = %arg0, %a1 : i32 = %arg1) stall(%stall) clock(%clock) reset(%reset) go(%go) entryEn(%s0_enable) -> (out: i32) {
      %add0 = comb.add %a0, %a1 : i32
      pipeline.stage ^bb1

    ^bb1(%s1_enable : i1):
      %add0_bb1 = pipeline.src %add0 : i32
      %a0_bb1 = pipeline.src %a0 : i32
      %add1 = comb.add %add0_bb1, %a0_bb1 : i32
      pipeline.stage ^bb2

    ^bb2(%s2_enable : i1):
      %add0_bb2 = pipeline.src %add0 : i32
      %add1_bb2 = pipeline.src %add1 : i32
      %add2 = comb.add %add1_bb2, %add0_bb2 : i32
      pipeline.return %add2 : i32
  }
  hw.output %out, %done : i32, i1
}
