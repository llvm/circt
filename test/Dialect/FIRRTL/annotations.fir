; RUN: circt-translate -import-firrtl -split-input-file %s  |FileCheck %s

; An Annotation without a target is promoted to a CircuitTarget Annotation. This
; represents a Scala FIRRTL Compiler `firrtl.annotations.NoTargetAnnotation`.
circuit Foo: %[[{"a":"a"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {raw_annotations = [{a = "a", target = "~Foo"}]}

; // -----

circuit Foo: %[[{"a":"a", target = "~"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {raw_annotations = [{a = "a", target = "~Foo"}]}

; // -----

; A legacy `firrtl.annotations.CircuitName` annotation becomes a CircuitTarget
; Annotation.
circuit Foo: %[[{"a":"a","target":"Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {raw_annotations = [{a = "a", target = "~Foo"}]}

; // -----

; A CircuitTarget Annotation is attached to the circuit.
circuit Foo: %[[{"a":"a","target":"~Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {raw_annotations = [{a = "a", target = "~Foo"}]}

; // -----

; A legacy `firrtl.annotations.ModuleName` annotation becomes a ModuleTarget
; Annotation
circuit Foo: %[[{"a":"a","target":"Foo.Foo"}]]
  module Foo:
    skip

    ; CHECK-LABEL: firrtl.circuit "Foo"
    ; CHECK: firrtl.module @Foo() attributes {raw_annotations = [{a = "a", target = "~"}]}
