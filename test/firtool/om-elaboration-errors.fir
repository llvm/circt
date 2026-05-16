; RUN: firtool %s -verify-diagnostics --output-final-mlir - -o /dev/null --strip-fir-debug-info=false

FIRRTL version 6.0.0
circuit OMElaborationErrors:
  class Child:
    output cond : Bool
    propassign cond, Bool(false)

  public module OMElaborationErrors:
    object child_obj of Child
    ; expected-error @below {{OM property assertion failed: must hold}}
    propassert child_obj.cond, "must hold"
