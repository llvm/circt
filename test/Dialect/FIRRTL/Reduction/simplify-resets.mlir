// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-simplify-resets | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// CHECK-LABEL: firrtl.circuit "ResetTypeInPorts"
firrtl.circuit "ResetTypeInPorts" {
  // CHECK: firrtl.module @ResetTypeInPorts(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %value: !firrtl.uint<8>)
  firrtl.module @ResetTypeInPorts(in %clock: !firrtl.clock, in %reset: !firrtl.reset, in %value: !firrtl.uint<8>) {
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    %reg = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.reset, !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInResults"
firrtl.circuit "ResetTypeInResults" {
  // CHECK: firrtl.module @ResetTypeInResults(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>)
  firrtl.module @ResetTypeInResults(in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
    // CHECK-NEXT: %wire = firrtl.wire : !firrtl.uint<1>
    %wire = firrtl.wire : !firrtl.reset
    // CHECK-NEXT: firrtl.matchingconnect %wire, %reset : !firrtl.uint<1>
    firrtl.matchingconnect %wire, %reset : !firrtl.reset
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInBlockArgs"
firrtl.circuit "ResetTypeInBlockArgs" {
  // CHECK: firrtl.module @ResetTypeInBlockArgs(in %cond: !firrtl.uint<1>, in %reset: !firrtl.uint<1>)
  firrtl.module @ResetTypeInBlockArgs(in %cond: !firrtl.uint<1>, in %reset: !firrtl.reset) {
    // CHECK-NEXT: firrtl.when %cond : !firrtl.uint<1> {
    firrtl.when %cond : !firrtl.uint<1> {
      // CHECK-NEXT: %wire = firrtl.wire : !firrtl.uint<1>
      %wire = firrtl.wire : !firrtl.reset
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInInstances"
firrtl.circuit "ResetTypeInInstances" {
  // CHECK: firrtl.module @ResetTypeInInstances(in %reset: !firrtl.uint<1>)
  firrtl.module @ResetTypeInInstances(in %reset: !firrtl.reset) {
    // CHECK-NEXT: %child_reset = firrtl.instance child @Child(in reset: !firrtl.uint<1>)
    %child_reset = firrtl.instance child @Child(in reset: !firrtl.reset)
    // CHECK-NEXT: firrtl.matchingconnect %child_reset, %reset : !firrtl.uint<1>
    firrtl.matchingconnect %child_reset, %reset : !firrtl.reset
  }

  // CHECK: firrtl.module private @Child(in %reset: !firrtl.uint<1>)
  firrtl.module private @Child(in %reset: !firrtl.reset) {
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInExtModule"
firrtl.circuit "ResetTypeInExtModule" {
  // CHECK: firrtl.module @ResetTypeInExtModule(in %reset: !firrtl.uint<1>)
  firrtl.module @ResetTypeInExtModule(in %reset: !firrtl.reset) {
    // CHECK-NEXT: %ext_reset = firrtl.instance ext @Ext(in reset: !firrtl.uint<1>)
    %ext_reset = firrtl.instance ext @Ext(in reset: !firrtl.reset)
    // CHECK-NEXT: firrtl.matchingconnect %ext_reset, %reset : !firrtl.uint<1>
    firrtl.matchingconnect %ext_reset, %reset : !firrtl.reset
  }

  // CHECK: firrtl.extmodule @Ext(in reset: !firrtl.uint<1>)
  firrtl.extmodule @Ext(in reset: !firrtl.reset)
}

// CHECK-LABEL: firrtl.circuit "MultipleResetTypes"
firrtl.circuit "MultipleResetTypes" {
  // CHECK: firrtl.module @MultipleResetTypes(in %reset1: !firrtl.uint<1>, in %reset2: !firrtl.uint<1>, out %reset_out: !firrtl.uint<1>)
  firrtl.module @MultipleResetTypes(in %reset1: !firrtl.reset, in %reset2: !firrtl.reset, out %reset_out: !firrtl.reset) {
    // CHECK-NEXT: %wire1 = firrtl.wire : !firrtl.uint<1>
    %wire1 = firrtl.wire : !firrtl.reset
    // CHECK-NEXT: %wire2 = firrtl.wire : !firrtl.uint<1>
    %wire2 = firrtl.wire : !firrtl.reset
    // CHECK-NEXT: firrtl.matchingconnect %wire1, %reset1 : !firrtl.uint<1>
    firrtl.matchingconnect %wire1, %reset1 : !firrtl.reset
    // CHECK-NEXT: firrtl.matchingconnect %wire2, %reset2 : !firrtl.uint<1>
    firrtl.matchingconnect %wire2, %reset2 : !firrtl.reset
    // CHECK-NEXT: firrtl.matchingconnect %reset_out, %wire1 : !firrtl.uint<1>
    firrtl.matchingconnect %reset_out, %wire1 : !firrtl.reset
  }
}

// CHECK-LABEL: firrtl.circuit "RemoveResetAnnotations"
firrtl.circuit "RemoveResetAnnotations" {
  // CHECK: firrtl.module @RemoveResetAnnotations(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>)
  // CHECK-NOT: circt.FullResetAnnotation
  // CHECK-NOT: circt.ExcludeFromFullResetAnnotation
  // CHECK-NOT: sifive.enterprise.firrtl.FullAsyncResetAnnotation
  // CHECK-NOT: sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation
  firrtl.module @RemoveResetAnnotations(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.reset [{class = "circt.FullResetAnnotation", resetType = "async"}]
  ) attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    // CHECK: %wire = firrtl.wire
    // CHECK-NOT: circt.FullResetAnnotation
    %wire = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.reset
  }
  // CHECK: firrtl.module private @Child
  // CHECK-NOT: sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation
  firrtl.module private @Child() attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
  }
}
