; RUN: firtool --parse-only --subcircuit-dir %S/subcircuits -split-input-file %s | FileCheck %s

circuit Top: %[[{"Top":"Top", "target":"~Top|Top"},
                {"a":"a", "target":"~Top|Top>a"},
                {"b":"b", "target":"~Top|Top>b"},
                {"c":"c", "target":"~Top|Top>c"}]]
  module Top:
    wire a: UInt<1>
    wire b: UInt<1>
    wire c: UInt<1>

; CHECK-LABEL: firrtl.module @Foo
; CHECK-SAME: annotations = [{Foo = "Foo"}]
; CHECK: %a = firrtl.wire
; CHECK-SAME: annotations = [{a = "a"}]
; CHECK: %b = firrtl.wire
; CHECK-SAME: annotations = [{b = "b"}]
; CHECK: %c = firrtl.wire
; CHECK-SAME: annotations = [{c = "c"}]

; CHECK-LABEL: firrtl.module @FooBar

; CHECK-LABEL: firrtl.module @Top
; CHECK-SAME: annotations = [{Top = "Top"}]
; CHECK: %a = firrtl.wire
; CHECK-SAME: annotations = [{a = "a"}]
; CHECK: %b = firrtl.wire
; CHECK-SAME: annotations = [{b = "b"}]
; CHECK: %c = firrtl.wire
; CHECK-SAME: annotations = [{c = "c"}]
