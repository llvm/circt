// RUN: circt-opt %s --arc-infer-context --split-input-file --verify-diagnostics

// A public function cannot gain an inferred context argument, since its
// signature is part of the module's external interface.
// expected-error @below {{Cannot infer an Arc context through a public function. A context argument must be provided explicitly.}}
func.func @publicDirect() -> i64 {
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}

// -----

// The context requirement propagates to callers, so a public function that
// transitively reaches a context-needing function is rejected too.
func.func private @leaf() -> i64 {
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}

// expected-error @below {{Cannot infer an Arc context through a public function. A context argument must be provided explicitly.}}
func.func @publicTransitive() -> i64 {
  %r = func.call @leaf() : () -> i64
  return %r : i64
}
