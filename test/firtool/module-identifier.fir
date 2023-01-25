; RUN: firtool %s --format=fir --parse-only -verify-diagnostics

; Test parsing of 'module' as an identifier for instance names

; CHECK-LABEL: moduleIdentifier
circuit Foo:
  module Bar:
      skip

  module Foo:
    inst module of Bar
