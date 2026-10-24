; RUN: firtool %s --parse-only | FileCheck %s --check-prefix=CREATE
; RUN: firtool %s --ir-hw | FileCheck %s --check-prefix=OM

FIRRTL version 6.0.0
circuit ObjectCreate:
  class PassThrough:
    input str: String
    output result: String
    propassign result, str

  public module ObjectCreate:
    output result: String
    ; CREATE-LABEL: firrtl.module @ObjectCreate
    ; CREATE-NEXT: %[[STR:.*]] = firrtl.string "hello"
    ; CREATE-NEXT: %obj = firrtl.object.create(%[[STR]]) : (!firrtl.string) -> !firrtl.class<@PassThrough(in str: !firrtl.string, out result: !firrtl.string)>
    ; CREATE-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %obj[result]
    ; CREATE-NEXT: firrtl.propassign %result, %[[RESULT]] : !firrtl.string
    ; CREATE-NEXT: }
    ; OM: om.class @ObjectCreate_Class
    ; OM: om.object @PassThrough
    object obj of PassThrough
    propassign obj.str, String("hello")
    propassign result, obj.result
