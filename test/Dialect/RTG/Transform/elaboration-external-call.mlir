// RUN: circt-opt --rtg-elaborate=seed=0 --split-input-file --verify-diagnostics %s

func.func private @external(%value: index) -> index

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

rtg.test @external_call() {
  %value = index.constant 41
  // expected-error @below {{cannot elaborate external function call}}
  %result = func.call @external(%value) : (index) -> index
  rtg.yield
}

// -----

func.func @identity(%value: index) -> index {
  return %value : index
}

rtg.target @target : !rtg.dict<> {
  rtg.yield
}

rtg.test @indirect_call() {
  %callee = func.constant @identity : (index) -> index
  %value = index.constant 41
  // expected-error @below {{cannot elaborate indirect function call}}
  %result = func.call_indirect %callee(%value) : (index) -> index
  rtg.yield
}
