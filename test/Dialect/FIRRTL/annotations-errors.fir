; RUN: circt-translate -import-firrtl -verify-diagnostics --split-input-file %s

; COM: A legacy `firrtl.annotations.ComponentName` is unsupported.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: Unsupported target (not a CircuitTarget or ModuleTarget) */ "Foo.Foo.a"}}
circuit Foo: %[[{"a":"a","target":"Foo.Foo.a"}]]
  module Foo:
    skip

; // -----

; COM: A ReferenceTarget Annotation is unsupported.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{"target": /* error: Unsupported target (not a CircuitTarget or ModuleTarget) */ "~Foo|Foo>a"}}
circuit Foo: %[[{"a":"a","target":"~Foo|Foo>a"}]]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is not an array.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array, but found something else. */}}
circuit Foo: %[{"a":"a"}]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is not an array of objects.

; expected-error @+2 {{Invalid/unsupported annotation format}}
; expected-note @+1 {{/* error: Expected annotations to be an array of objects, but found an array of something else. */}}
circuit Foo: %[[{"a":"a"},[{"b":"b"}]]]
  module Foo:
    skip

; // -----

; COM: Annotations JSON is invalid. (There's a trailing comma.)

; expected-error @+2 {{Failed to parse JSON Annotations}}
; expected-note @+1 {{}}
circuit Foo: %[[{"a":"a"},]]]
  module Foo:
    skip
