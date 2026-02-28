; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; Annotations JSON is not an array.

FIRRTL version 4.0.0
; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array, but found something else. */}}
circuit Foo: %[{"a":"a"}]
  public module Foo:
    skip

; // -----

; Annotations JSON is not an array of objects.

FIRRTL version 4.0.0
; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array of objects, but found an array of something else. */}}
circuit Foo: %[[{"a":"a"},[{"b":"b"}]]]
  public module Foo:
    skip

; // -----

; Annotations JSON is invalid. (There's a trailing comma.)

FIRRTL version 4.0.0
; expected-error @+2 {{Failed to parse JSON Annotations}}
; expected-note @+1 {{}}
circuit Foo: %[[{"a":"a"},]]]
  public module Foo:
    skip
