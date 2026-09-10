; RUN: firtool -disable-annotation-unknown %s | FileCheck %s

FIRRTL version 5.2.0
circuit Foo: %[[
  {
    "class": "Unknown"
  }
]]

  ; CHECK: module Foo(
  public module Foo:
    input a: UInt<1>
