// RUN: circt-opt --sim-lower-dpi-func %s --verify-diagnostics --split-input-file

// expected-error @below {{references existing func.func @foo_c with incompatible type '() -> ()'; expected '(!llvm.ptr) -> ()'}}
sim.func.dpi @foo(out result: i32) attributes {verilogName = "foo_c"}
func.func @foo_c() {
  func.return
}

hw.module @top() {
  hw.output
}
