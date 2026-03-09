// RUN: circt-opt --split-input-file -lower-calyx-to-hw %s | FileCheck %s

// Verify that convertPipelineOp converts the i1 clock to seq.clock
// before creating registers. This is a regression test for a bug where
// convertPipelineOp passed i1 directly to seq.compreg (which requires
// seq.clock), while the RegisterOp case was already correctly using
// seq.to_clock.

// CHECK-LABEL: hw.module @main
// CHECK-DAG:   %[[MU_CLK_VAL:.+]] = sv.read_inout %mu_clk
// CHECK-DAG:   %[[SEQ_CLK:.+]] = seq.to_clock %[[MU_CLK_VAL]]
// CHECK-DAG:   seq.compreg sym @mu_done %{{.+}}, %[[SEQ_CLK]]
// CHECK-DAG:   seq.compreg.ce sym @mu_out %{{.+}}, %[[SEQ_CLK]]
module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%in0: i32, %in1: i32, %clk: i1 {clk}, %reset: i1 {reset}, %go: i1 {go}) -> (%out0: i32, %done: i1 {done}) {
    %true = hw.constant true
    %mu.clk, %mu.reset, %mu.go, %mu.left, %mu.right, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    calyx.wires {
      calyx.assign %mu.clk = %clk : i1
      calyx.assign %mu.reset = %reset : i1
      calyx.assign %mu.go = %go : i1
      calyx.assign %mu.left = %in0 : i32
      calyx.assign %mu.right = %in1 : i32
      calyx.assign %out0 = %mu.out : i32
      calyx.assign %done = %mu.done : i1
    }
    calyx.control {}
  }
}
