// RUN: circt-opt %s --arc-infer-context | FileCheck %s

// An `arc.inferred_context` in the body of an `arc.model` resolves to the
// context derived from the model's storage argument.
// CHECK-LABEL: arc.model @ModelBody
arc.model @ModelBody io !hw.modty<> {
// CHECK-NEXT: ^bb0(%arg0: !arc.storage):
^bb0(%arg0: !arc.storage):
  // CHECK-NEXT: [[CTX:%.+]] = arc.as_context %arg0 : !arc.storage
  // CHECK-NEXT: arc.current_time [[CTX]]
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
}

// The context is derived at the start of the region even when the
// `arc.inferred_context` is not the first operation.
// CHECK-LABEL: arc.model @ModelBodyLate
arc.model @ModelBodyLate io !hw.modty<> {
// CHECK-NEXT: ^bb0(%arg0: !arc.storage):
^bb0(%arg0: !arc.storage):
  // CHECK-NEXT: [[CTX:%.+]] = arc.as_context %arg0 : !arc.storage
  // CHECK-NEXT: [[T:%.+]] = hw.constant 0
  // CHECK-NEXT: arc.set_next_wakeup [[CTX]], [[T]]
  %t = hw.constant 0 : i64
  %ctx = arc.inferred_context
  arc.set_next_wakeup %ctx, %t
}

// An `arc.inferred_context` in the body of an `arc.sim.instantiate` resolves to
// the context derived from the instance handle. The enclosing (public) function
// is not itself flagged as needing a context argument, since the instance
// provides the context.
arc.model @sim_test io !hw.modty<> {
^bb0(%arg0: !arc.storage):
}
// CHECK-LABEL: func.func @InstanceBody() {
func.func @InstanceBody() {
  // CHECK: arc.sim.instantiate @sim_test as [[INST:%.+]] {
  arc.sim.instantiate @sim_test as %model {
    // CHECK-NEXT: [[CTX:%.+]] = arc.as_context [[INST]] : !arc.sim.instance<@sim_test>
    // CHECK-NEXT: arc.current_time [[CTX]]
    %ctx = arc.inferred_context
    %t = arc.current_time %ctx
  }
  return
}

// A private function containing an `arc.inferred_context` gains a trailing
// context argument, and each caller threads its own context in.
// CHECK-LABEL: func.func private @needsCtx
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @needsCtx() -> i64 {
  // CHECK-NEXT: [[T:%.+]] = arc.current_time %arg0
  // CHECK-NEXT: return [[T]]
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}

// CHECK-LABEL: arc.model @Caller
arc.model @Caller io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: [[CTX:%.+]] = arc.as_context %arg0
  // CHECK-NEXT: func.call @needsCtx([[CTX]]) : (!arc.context) -> i64
  %r = func.call @needsCtx() : () -> i64
}

// A function that already has a context argument keeps its signature and
// resolves `arc.inferred_context` to that argument.
// CHECK-LABEL: func.func private @hasCtx
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @hasCtx(%c: !arc.context) -> i64 {
  // CHECK: arc.current_time %arg0
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}

// A context-providing function is resolved even when it is public: no argument
// has to be added, so the public-function restriction does not apply.
// CHECK-LABEL: func.func @pubHasCtx
// CHECK-SAME:    (%arg0: !arc.context)
func.func @pubHasCtx(%c: !arc.context) -> i64 {
  // CHECK: arc.current_time %arg0
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}

// The context threads transitively through a call chain: the model provides it,
// and every private function on the path gains a context argument.
// CHECK-LABEL: func.func private @chainLeaf
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @chainLeaf() -> i64 {
  // CHECK: arc.current_time %arg0
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}
// CHECK: func.func private @chainMid(%arg0: !arc.context)
func.func private @chainMid() -> i64 {
  // CHECK-NEXT: [[R:%.+]] = call @chainLeaf(%arg0) : (!arc.context) -> i64
  // CHECK-NEXT: return [[R]]
  %r = func.call @chainLeaf() : () -> i64
  return %r : i64
}
// CHECK: arc.model @ChainCaller
arc.model @ChainCaller io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: [[CTX:%.+]] = arc.as_context %arg0
  // CHECK-NEXT: func.call @chainMid([[CTX]]) : (!arc.context) -> i64
  %r = func.call @chainMid() : () -> i64
}

// An `arc.inferred_context` nested in a non-isolated region (e.g. `scf.if`) of a
// function still resolves to the threaded context argument.
// CHECK-LABEL: func.func private @nestedRegion
// CHECK-SAME:    (%arg0: i1, %arg1: !arc.context)
func.func private @nestedRegion(%cond: i1) {
  // CHECK-NEXT: scf.if
  // CHECK-NEXT:   arc.current_time %arg1
  scf.if %cond {
    %ctx = arc.inferred_context
    %t = arc.current_time %ctx
  }
  return
}
// CHECK: arc.model @NestedCaller
arc.model @NestedCaller io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: [[CTX:%.+]] = arc.as_context %arg0
  // CHECK: func.call @nestedRegion(%{{.+}}, [[CTX]]) : (i1, !arc.context) -> ()
  %c = hw.constant true
  func.call @nestedRegion(%c) : (i1) -> ()
}

// A callee shared between two models gets a single context argument; each model
// passes its own context at its call site.
// CHECK-LABEL: func.func private @shared
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @shared() -> i64 {
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}
// CHECK: arc.model @ShareA
arc.model @ShareA io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: [[CTX1:%.+]] = arc.as_context %arg0
  // CHECK-NEXT: func.call @shared([[CTX1]])
  %r = func.call @shared() : () -> i64
}
// CHECK: arc.model @ShareB
arc.model @ShareB io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: [[CTX2:%.+]] = arc.as_context %arg0
  // CHECK-NEXT: func.call @shared([[CTX2]])
  %r = func.call @shared() : () -> i64
}

// Multiple `arc.inferred_context` ops in one function all resolve to the same
// context argument.
// CHECK-LABEL: func.func private @multiInfer
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @multiInfer() -> i64 {
  // CHECK-NEXT: [[A:%.+]] = arc.current_time %arg0
  // CHECK-NEXT: [[B:%.+]] = arc.current_time %arg0
  // CHECK-NEXT: comb.add [[A]], [[B]]
  %c1 = arc.inferred_context
  %a = arc.current_time %c1
  %c2 = arc.inferred_context
  %b = arc.current_time %c2
  %s = comb.add %a, %b : i64
  return %s : i64
}
// CHECK: arc.model @MultiCaller
arc.model @MultiCaller io !hw.modty<> {
^bb0(%arg0: !arc.storage):
  // CHECK: func.call @multiInfer
  %r = func.call @multiInfer() : () -> i64
}

// A function that already has a context argument threads that same argument into
// the context-needing functions it calls.
// CHECK-LABEL: func.func private @provideLeaf
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @provideLeaf() -> i64 {
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}
// CHECK: func.func private @provideMid(%arg0: !arc.context)
func.func private @provideMid(%c: !arc.context) -> i64 {
  // CHECK: call @provideLeaf(%arg0) : (!arc.context) -> i64
  %r = func.call @provideLeaf() : () -> i64
  return %r : i64
}

// A cycle in the call graph where one function needs a context: both functions
// gain the argument and it is threaded around the cycle.
// CHECK-LABEL: func.func private @recA
// CHECK-SAME:    (%arg0: i32, %arg1: !arc.context)
func.func private @recA(%n: i32) -> i64 {
  // CHECK: arc.current_time %arg1
  // CHECK: call @recB(%arg0, %arg1) : (i32, !arc.context) -> i64
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  %r = func.call @recB(%n) : (i32) -> i64
  return %t : i64
}
// CHECK: func.func private @recB(%arg0: i32, %arg1: !arc.context)
func.func private @recB(%n: i32) -> i64 {
  // CHECK: call @recA(%arg0, %arg1) : (i32, !arc.context) -> i64
  %r = func.call @recA(%n) : (i32) -> i64
  return %r : i64
}

// A private context-needing function that is never called still gains a context
// argument (there is simply no call site to update).
// CHECK-LABEL: func.func private @uncalled
// CHECK-SAME:    (%arg0: !arc.context)
func.func private @uncalled() -> i64 {
  // CHECK-NEXT: arc.current_time %arg0
  %ctx = arc.inferred_context
  %t = arc.current_time %ctx
  return %t : i64
}
