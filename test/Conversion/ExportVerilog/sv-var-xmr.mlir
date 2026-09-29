// RUN: circt-opt %s --export-verilog | FileCheck %s

// ThisVerify that an sv.var with an inner symbol can be the target of an XMR
// hierpath and that ExportVerilog emits the expected hierarchical reference.

// Note: The correct handles for sv.xmr.ref and sv.read_inout will be updated soon.

// CHECK: module test_xmr_var();
// CHECK: var logic [31:0] v;
// CHECK: module test_xmr_consumer(
// CHECK: test_xmr_var var_inst ();
// CHECK: assign out = test_xmr_consumer.var_inst.v;

module {
  hw.hierpath @v_path [
      @test_xmr_consumer::@var_inst,
      @test_xmr_var::@v_sym
  ]

  hw.module @test_xmr_var() {
    %v = sv.var sym @v_sym : !sv.var<i32>
  }

  hw.module @test_xmr_consumer(out out : i32) {
    hw.instance "var_inst" sym @var_inst @test_xmr_var() -> ()
  
    %x = sv.xmr.ref @v_path : !hw.inout<i32>
    %value = sv.read_inout %x : !hw.inout<i32>
    hw.output %value : i32
  }
}
