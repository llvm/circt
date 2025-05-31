// REQUIRES: iverilog,cocotb

// RUN: circt-opt %s -pipeline-explicit-regs -lower-pipeline-to-hw -lower-seq-to-sv -sv-trace-iverilog -export-verilog \
// RUN:     -o %t.mlir > %t.sv

// RUN: circt-cocotb-driver.py --objdir=%T --topLevel=nonstallable_test1 \
// RUN:     --pythonModule=nonstallable_test1 --pythonFolder="%S,%S/.." %t.sv 2>&1 | FileCheck %s


// CHECK: ** TEST
// CHECK: ** TESTS=[[N:.*]] PASS=[[N]] FAIL=0 SKIP=0

hw.module @nonstallable_test1(in %arg0: i32, in %go: i1, in %clock: !seq.clock, in %reset: i1, in %stall: i1, out out: i32, out done: i1) {
  %out, %done = pipeline.scheduled "nonstallable_test1"(%a0 : i32 = %arg0)
      stall(%stall) clock(%clock) reset(%reset) go(%go) entryEn(%s0_enable)
        {stallability = [true, false, false, true, true]} -> (out : i32) {
    pipeline.stage ^bb1
  ^bb1(%s1_enable: i1):
    pipeline.stage ^bb2
  ^bb2(%s2_enable: i1):
    pipeline.stage ^bb3
  ^bb3(%s3_enable: i1):
    pipeline.stage ^bb4
  ^bb4(%s4_enable: i1):
    pipeline.stage ^bb5
  ^bb5(%s5_enable: i1):
    %a0_bb5 = pipeline.src %a0 : i32
    pipeline.return %a0_bb5 : i32
  }
  hw.output %out, %done : i32, i1
}
