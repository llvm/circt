circuit Foo:
  module Foo:
    wire a: UInt<1>
    wire b: UInt<1>
    wire c: UInt<1>
    inst FooBar of FooBar

  module FooBar:
    wire a: UInt<1>
    wire b: UInt<1>
    wire c: UInt<1>
