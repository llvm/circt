; RUN: circt-translate --import-firrtl --mlir-print-op-generic --split-input-file %s | FileCheck %s

; A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.
circuit Foo: %[[{"a":"a","target":"~Foo|Bar>bar"},{"b":"b","target":"Foo.Foo.foo"}]]
  extmodule Bar:
    input bar: UInt<1>
  module Foo:
    input foo: UInt<1>
    inst bar of Bar
    bar.bar <= foo

; CHECK-LABEL: "firrtl.extmodule"() ({
; CHECK: }) {
; CHECK-SAME: portAnnotations = {{['[']['[']}}{a = "a"}]]
; CHECK-SAME: sym_name = "Bar"

; CHECK-LABEL: "firrtl.module"() ({
; CHECK: }) {
; CHECK-SAME: portAnnotations = {{['[']['[']}}{b = "b"}]]
; CHECK-SAME: sym_name = "Foo"
