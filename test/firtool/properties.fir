; RUN: firtool %s 2>&1 | FileCheck %s

; Check that we can successfully compile this to Verilog without crashing.

; CHECK: module Prop()
FIRRTL version 3.1.0
circuit Prop:
  module Prop:
    output y : String
    propassign y, String("test")
