; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

; COM: An Annotation without a target is promoted to a CircuitTarget
; COM: Annotation. This represents a Scala FIRRTL Compiler
; COM: `firrtl.annotations.NoTargetAnnotation`.
circuit Foo: %[[{"a":"a"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A legacy `firrtl.annotations.CircuitName` annotation becomes a
; COM: CircuitTarget Annotation.
circuit Foo: %[[{"a":"a","target":"Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A CircuitTarget Annotation is attached to the circuit.
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = "a"}]}

; // -----

; COM: A legacy `firrtl.annotations.ModuleName` annotation becomes a
; COM: ModuleTarget Annotation
circuit Foo: %[[{"a":"a","target":"Foo.Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}

; // -----

; COM: A ModuleTarget Annotation is attached to the correct module.
circuit Foo: %[[{"a":"a","target":"~Foo|Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.module @Foo() attributes {annotations = [{a = "a"}]}
