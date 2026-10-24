; RUN: circt-translate -import-firrtl %s | FileCheck %s --check-prefix=PARSER
; RUN: firtool %s --parse-only | FileCheck %s --check-prefix=CREATE
; RUN: firtool %s --ir-hw | FileCheck %s --check-prefix=OM
; RUN: env CIRCT_DISABLE_LOWER_OBJECTS=1 firtool %s --parse-only | FileCheck %s --check-prefix=DISABLED

FIRRTL version 6.0.0
circuit ObjectCreate:
  class PassThrough:
    input str: String
    output result: String
    propassign result, str

  public module ObjectCreate:
    output result: String
    ; PARSER: firrtl.object @PassThrough
    ; PARSER: firrtl.propassign
    ; CREATE-LABEL: firrtl.module @ObjectCreate
    ; CREATE-NEXT: %[[STR:.*]] = firrtl.string "hello"
    ; CREATE-NEXT: %obj = firrtl.object.create(%[[STR]]) : (!firrtl.string) -> !firrtl.class<@PassThrough(in str: !firrtl.string, out result: !firrtl.string)>
    ; CREATE-NEXT: %[[RESULT:.*]] = firrtl.object.subfield %obj[result]
    ; CREATE-NEXT: firrtl.propassign %result, %[[RESULT]] : !firrtl.string
    ; CREATE-NEXT: }
    ; DISABLED-LABEL: firrtl.module @ObjectCreate
    ; DISABLED: %obj = firrtl.object @PassThrough
    ; DISABLED: firrtl.propassign
    ; OM: om.class @ObjectCreate_Class
    ; OM: om.object @PassThrough
    object obj of PassThrough
    propassign obj.str, String("hello")
    propassign result, obj.result
