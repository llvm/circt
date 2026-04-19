// RUN: circt-verilog %s | FileCheck %s
// REQUIRES: slang
// Internal issue in Slang about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

module SubB();
  logic local_val;
  assign local_val = 1'b1;
endmodule

module SubC(input logic in_val);
endmodule

module SubD();
  logic local_val;
  assign local_val = 1'b1;
endmodule

// Multiple instantiations with same name.
module Top();
  SubB b_inst();
  SubD d_inst();
  SubC c_inst1(.in_val(b_inst.local_val));
  SubC c_inst2(.in_val(d_inst.local_val));
endmodule

// CHECK: hw.module private @SubB(out local_val : !llhd.ref<i1>)
// CHECK: hw.module private @SubC(in %in_val : i1)
// CHECK: hw.module private @SubD(out local_val : !llhd.ref<i1>)

// CHECK: hw.module @Top() {
// CHECK:   %[[REF1:.+]] = hw.instance "b_inst" @SubB() -> (local_val: !llhd.ref<i1>)
// CHECK:   %[[REF2:.+]] = hw.instance "d_inst" @SubD() -> (local_val: !llhd.ref<i1>)
// CHECK:   %[[VAL1:.+]] = llhd.prb %[[REF1]] : i1
// CHECK:   hw.instance "c_inst1" @SubC(in_val: %[[VAL1]]: i1) -> ()
// CHECK:   %[[VAL2:.+]] = llhd.prb %[[REF2]] : i1
// CHECK:   hw.instance "c_inst2" @SubC(in_val: %[[VAL2]]: i1) -> ()
// CHECK: }

// Self-referencing hierarchical name
module SelfRef();
  logic self_z;
  logic self_w;
  assign SelfRef.self_w = SelfRef.self_z;
endmodule
// CHECK: hw.module @SelfRef()

// Multi-instance deduplication and shared access.
module SharedChild();
  logic child_val;
  assign child_val = 1'b0;
endmodule

module ParentInst();
  SharedChild child();
endmodule

module DeduplicationTest();
  ParentInst p1();
  ParentInst p2();
  logic read_p1;
  logic read_p2;
  assign read_p1 = p1.child.child_val;
  assign read_p2 = p2.child.child_val;
endmodule
// CHECK: hw.module private @SharedChild(out child_val : !llhd.ref<i1>)
// CHECK: hw.module private @ParentInst(out child.child_val : !llhd.ref<i1>) {
// CHECK:   %[[CHILD_REF:.+]] = hw.instance "child" @SharedChild() -> (child_val: !llhd.ref<i1>)
// CHECK:   hw.output %[[CHILD_REF]] : !llhd.ref<i1>
// CHECK: }

// CHECK: hw.module @DeduplicationTest() {
// CHECK:   %[[P1_REF:.+]] = hw.instance "p1" @ParentInst() -> (child.child_val: !llhd.ref<i1>)
// CHECK:   %[[P2_REF:.+]] = hw.instance "p2" @ParentInst() -> (child.child_val: !llhd.ref<i1>)
// CHECK:   hw.output
// CHECK: }

// Upward reference to root.
module TopRoot();
  logic root_val;
  assign root_val = 1'b1;
  MiddleInst mid();
endmodule

module MiddleInst();
  LeafInst leaf();
endmodule

module LeafInst();
  logic leaf_val;
  assign leaf_val = TopRoot.root_val;
endmodule
// CHECK: hw.module @TopRoot() {
// CHECK:   %[[ROOT_VAL:.+]] = llhd.sig %false : i1
// CHECK:   hw.instance "mid" @MiddleInst(TopRoot.root_val: %[[ROOT_VAL]]: !llhd.ref<i1>) -> ()
// CHECK: }

// CHECK: hw.module private @MiddleInst(in %TopRoot.root_val : !llhd.ref<i1>) {
// CHECK:   hw.instance "leaf" @LeafInst(TopRoot.root_val: %TopRoot.root_val: !llhd.ref<i1>) -> ()
// CHECK: }

// CHECK: hw.module private @LeafInst(in %TopRoot.root_val : !llhd.ref<i1>) {
// CHECK: }
