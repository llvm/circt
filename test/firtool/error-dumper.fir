; `verify-diagnostics` does not work with external locators so use `not`
; RUN: not firtool %s  --output-error-diagnostics=- --split-input-file -fuse-info-locators | FileCheck %s
FIRRTL
; CHECK: [{"location":[{"column":1,"file":"{{.+}}","line":{{[0-9]+}}}],"message":"expected version after 'FIRRTL'"}]
circuit test :
  module test :
    skip

; // -----
; Expand when
; CHECK:      [{"location":[{"column":1,"file":"test.scala","line":1},{"column":12,"file":"{{.+}}","line":{{[0-9]+}}}],"message":"port \"a\" not fully initialized in \"test\""},{
; CHECK-SAME:   "location":[{"column":12,"file":"{{.+}}","line":{{[0-9]+}}}],"message":"port \"b\" not fully initialized in \"test\""}]
FIRRTL version 5.0.0
circuit test :
  public module test :
    output a : UInt<1> @[test.scala 1:1]
    output b : UInt<1>
