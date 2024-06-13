; RUN: firtool %s --verbose-pass-executions 2>&1 | FileCheck %s
;
; CHECK: [firtool] Running

FIRRTL version 4.0.0
circuit Empty:
  public module Empty:
