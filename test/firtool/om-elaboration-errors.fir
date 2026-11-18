; RUN: firtool %s -verify-diagnostics --output-final-mlir - -o /dev/null --strip-fir-debug-info=false

FIRRTL version 7.0.0
circuit OMElaborationErrors:
  class Child:
    output cond : Bool
    propassign cond, Bool(false)

  class Options:
    output none : List<Integer>
    output some : List<Integer>
    propassign none, List<Integer>()
    propassign some, List<Integer>(Integer(42))

  public module OMElaborationErrors:
    object child_obj of Child
    ; expected-error @below {{OM property assertion failed: must hold}}
    propassert child_obj.cond, "must hold"

    object options of Options
    propassert prop_eq(list_length(options.none), Integer(0)), "empty option"
    propassert prop_eq(list_length(options.some), Integer(1)), "present option"

    ; expected-error @below {{OM property assertion failed: present option is not empty}}
    propassert prop_eq(list_length(options.some), Integer(0)), "present option is not empty"
