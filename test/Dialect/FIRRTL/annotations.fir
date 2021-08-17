; RUN: circt-translate -import-firrtl -split-input-file %s  |FileCheck %s

; All types of JSON values should work
circuit Foo: %[[
  {
    "class" : "circt.testNT",
    "string": "a",
     "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null": null,
    "object": {
      "foo": "bar"
    },
    "array": [1, 2, 3]
  }
]]
  module Foo:
    skip
    ; CHECK: module {
    ; CHECK-LABEL: firrtl.circuit "Foo" attributes {annotations =
    ; CHECK-SAME: array = [1, 2, 3]
    ; CHECK-SAME: boolean = true
    ; CHECK-SAME: float = 3.140
    ; CHECK-SAME: integer = 42
    ; CHECK-SAME: object = {foo = "bar"}
    ; CHECK-SAME: string = "a"

; // -----

; JSON escapes should work.
circuit Foo: %[[{"\"":"}]]","class":"circt.testNT"}]]
  module Foo:
    skip

    ; CHECK: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{"\22" = "}]]", class = "circt.testNT"}]}

; // -----

; JSON with a JSON-quoted string should be expanded.
circuit Foo: %[[{"a":"{\"b\":null}", "class": "circt.testNT"}]]
  module Foo:
    skip

    ; CHECK: module {
    ; CHECK: firrtl.circuit "Foo" attributes {annotations = [{a = {b}, class = "circt.testNT"}]}
