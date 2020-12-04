// RUN: circt-opt %s -split-input-file -verify-diagnostics

rtl.module @test_instance_exist_error() {
  // expected-error @+1 {{Symbol not found: @noexist.}}
  %b = sv.interface.instance : !sv.interface<@noexist>
}

// -----

module {
  rtl.module @foo () {  }
  // expected-error @+1 {{Symbol @foo is not an InterfaceOp.}}
  %b = sv.interface.instance : !sv.interface<@foo>
}
