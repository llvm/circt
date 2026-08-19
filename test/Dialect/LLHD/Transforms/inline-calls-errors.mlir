// RUN: circt-opt --llhd-inline-calls --verify-diagnostics --split-input-file %s

hw.module @CallInGraphRegion() {
  // expected-error @below {{function call cannot be inlined}}
  func.call @foo() : () -> ()
}

func.func @foo() {
  return
}
