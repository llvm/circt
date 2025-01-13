; RUN: firtool %s -ir-verilog | FileCheck %s

FIRRTL version 4.1.0

; COM: Check that we can even target an extmodule.
circuit Test : %[[
  {
    "class": "sifive.enterprise.firrtl.MarkDUTAnnotation",
    "target": "~Test|DUT"
  }
]]
  ; CHECK: hw.hierpath private [[DUT_NLA:@.+]] [@Test::[[DUT_SYM:@.+]]]
  public module Test :
    input in : UInt<1>
    output out : UInt<1>

    ; CHECK: hw.instance "dut" sym [[DUT_SYM]]
    inst dut of DUT

    connect dut.in, in
    connect out, dut.out

  extmodule DUT :
    input in : UInt<1>
    output out : UInt<1>

  ; COM: Check that metadata includes the dutModulePath pointing to Test::dut.
  ; CHECK: om.class @SiFive_Metadata(%basepath: !om.basepath) -> (dutModulePath
  ; CHECK-NEXT: om.path_create instance %basepath [[DUT_NLA]]
