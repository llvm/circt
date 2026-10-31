// RUN: circt-opt %s --verify-diagnostics --split-input-file

hw.module @Alu() {}

// expected-error @+2 {{'dbg.instance' must be a list of #dbg.instance}}
hw.module @NotInstanceAttr() {
  hw.instance "a" @Alu() -> () {dbg.instance = "a"}
}

// -----

hw.module @Alu() {}

// expected-error @+2 {{'dbg.instance' path must hold strings}}
hw.module @PathNotStrings() {
  hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a", path = [1 : i32]>]}
}

// -----

// An ext key is a lowercase language name.

hw.module @Alu() {}

// expected-error @+2 {{ext key 'Chisel' is not a lowercase language name}}
hw.module @ExtBadLanguage() {
  hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a", ext = {Chisel = "{}"}>]}
}

// -----

hw.module @Alu() {}

// expected-error @+2 {{ext value of 'chisel' must be a string}}
hw.module @ExtNotString() {
  hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a", ext = {chisel = 1 : i32}>]}
}

// -----

hw.module @Alu() {}

// expected-error @+2 {{ext key '1x' is not a lowercase language name}}
hw.module @ExtDigitFirstKey() {
  hw.instance "a" @Alu() -> () {dbg.instance = [#dbg.instance<name = "a", ext = {"1x" = "{}"}>]}
}

// -----

// A scope of an inlined instance checks its ext the same way.

func.func @ScopeExtNotString() {
  // expected-error @+1 {{ext value of 'chisel' must be a string}}
  %0 = dbg.scope "a", "Alu" ext {chisel = 1 : i32}
  return
}
