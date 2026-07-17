// RUN: circt-opt --llhd-inline-calls --verify-diagnostics --split-input-file %s

hw.module @CallInGraphRegion() {
  // expected-error @below {{function call cannot be inlined}}
  func.call @foo() : () -> ()
}

func.func @foo() {
  return
}

// -----

hw.module @RecursiveSuspendingCalls() {
  llhd.process {
    func.call @foo() : () -> ()
    llhd.halt
  }
}

llhd.coroutine @tick() {
  llhd.return
}

// expected-note @below {{call target suspends execution and must be inlined}}
func.func @foo() {
  llhd.call_coroutine @tick() : () -> ()
  // expected-error @below {{recursive function call cannot be inlined}}
  call @foo() : () -> ()
  return
}
