; RUN: circt-translate --import-firrtl --split-input-file %s |circt-opt -firrtl-lower-annotations --mlir-print-op-generic | FileCheck %s

; A ReferenceTarget/ComponentName pointing at a module/extmodule port should work.
circuit Foo: %[[{"class":"circt.test","a":"a","target":"~Foo|Bar>bar"},{"class":"circt.test","b":"b","target":"Foo.Foo.foo"}]]
  extmodule Bar:
    input bar: UInt<1>
  module Foo:
    input foo: UInt<1>
    inst bar of Bar
    bar.bar <= foo

; CHECK-LABEL: "firrtl.extmodule"() ( {
; CHECK: }) {
; CHECK-SAME: portAnnotations = {{['[']['[']}}{a = "a", class = "circt.test"}]]
; CHECK-SAME: sym_name = "Bar"

; CHECK-LABEL: "firrtl.module"() ( {
; CHECK: }) {
; CHECK-SAME: portAnnotations = {{['[']['[']}}{b = "b", class = "circt.test"}]]
; CHECK-SAME: sym_name = "Foo"
