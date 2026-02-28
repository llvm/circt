// RUN: circt-opt --llhd-inline-calls --verify-diagnostics --split-input-file %s

hw.module @CallInGraphRegion() {
  // expected-error @below {{function call cannot be inlined}}
  func.call @foo() : () -> ()
}

func.func @foo() {
  return
}

// -----

hw.module @RecursiveCalls() {
  llhd.combinational {
    func.call @foo() : () -> ()
    llhd.yield
  }
}

func.func @foo() {
  call @bar() : () -> ()
  return
}

func.func @bar() {
  // expected-error @below {{recursive function call cannot be inlined}}
  call @foo() : () -> ()
  return
}
