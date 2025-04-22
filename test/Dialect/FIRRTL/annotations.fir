; RUN: circt-translate -import-firrtl -split-input-file %s | FileCheck %s

FIRRTL version 4.0.0
; All types of JSON values should work
circuit Foo: %[[
  {
    "class": "circt.testNT",
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
  public module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo"
    ; CHECK-SAME: rawAnnotations =
    ; CHECK-SAME: array = [1, 2, 3]
    ; CHECK-SAME: boolean = true
    ; CHECK-SAME: float = 3.140
    ; CHECK-SAME: integer = 42
    ; CHECK-SAME: object = {foo = "bar"}
    ; CHECK-SAME: string = "a"

; // -----

FIRRTL version 4.0.0
; JSON escapes should work.
circuit Foo: %[[{"class": "circt.testNT", "\"":"}]]"}]]
  public module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo"
    ; CHECK-SAME: rawAnnotations =

; // -----
FIRRTL version 4.0.0
; JSON with a JSON-quoted string should be expanded.
circuit Foo: %[[{"class":"circt.testNT","a":"{\"b\":null}"}]]
  public module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK: firrtl.circuit "Foo"
    ; CHECK-SAME: rawAnnotations = [{a = {b}, class = "circt.testNT"}]

; // -----
FIRRTL version 4.0.0
; JSON with a JSON-quoted number should stay as a number.
circuit Foo: %[[{"class":"circt.testNT","a":"0","b":0,"c":"\"0\""}]]
  public module Foo:
    skip

    ; CHECK-LABEL: module {
    ; CHECK:         firrtl.circuit "Foo"
    ; CHECK-SAME:      a = "0"
    ; CHECK-SAME:      b = 0
    ; CHECK-SAME:      c = "0"

; // -----
;
; A numeric "class" shouldn't crash the parser.

FIRRTL version 4.0.0
circuit Top : %[[
  {
    "class":"0",
    "target":"~Top|Top>a"
  }
]]
  public module Top :
    wire a: UInt<1>
    invalidate a

; CHECK-LABEL: firrtl.circuit "Top"


; // -----
FIRRTL version 4.0.0
; Test that a circuit with annotations that are supposed to be bypassed by
; parsing/scattering are properly moved onto the circuit under a
; "rawAnnotations" attribute.
circuit RawAnnotations: %[[
  {
    "class":"circt.test"
  },
  {
    "class":"circt.testNT"
  },
  {
    "class":"circt.missing"
  }
]]
  public module RawAnnotations:
    wire x: UInt<1>

    ; CHECK-LABEL: firrtl.circuit "RawAnnotations"
    ; CHECK-SAME:    rawAnnotations = [
    ; CHECK-SAME:      "circt.test"
    ; CHECK-SAME:      "circt.testNT"
    ; CHECK-SAME:      "circt.missing"
