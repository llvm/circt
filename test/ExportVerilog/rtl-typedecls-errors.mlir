// RUN: circt-translate %s -export-verilog -verify-diagnostics

// expected-error @+1 {{unable to resolve name for type reference}}
rtl.module @testTypeRef(
  %arg0: !rtl.typeref<@__rtl_typedecls::@foo>) {
}
