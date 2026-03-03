// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include firrtl-simplify-resets | FileCheck %s

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129

// CHECK-LABEL: firrtl.circuit "ResetTypeInPorts"
firrtl.circuit "ResetTypeInPorts" {
  // CHECK: firrtl.module @ResetTypeInPorts(in %clock: !firrtl.clock, in %reset: !firrtl.syncreset, in %value: !firrtl.uint<8>)
  firrtl.module @ResetTypeInPorts(in %clock: !firrtl.clock, in %reset: !firrtl.inferredreset, in %value: !firrtl.uint<8>) {
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.syncreset, !firrtl.uint<8>, !firrtl.uint<8>
    %reg = firrtl.regreset %clock, %reset, %value : !firrtl.clock, !firrtl.inferredreset, !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInResults"
firrtl.circuit "ResetTypeInResults" {
  // CHECK: firrtl.module @ResetTypeInResults(in %clock: !firrtl.clock, in %reset: !firrtl.syncreset)
  firrtl.module @ResetTypeInResults(in %clock: !firrtl.clock, in %reset: !firrtl.inferredreset) {
    // CHECK-NEXT: %wire = firrtl.wire : !firrtl.syncreset
    %wire = firrtl.wire : !firrtl.inferredreset
    // CHECK-NEXT: firrtl.matchingconnect %wire, %reset : !firrtl.syncreset
    firrtl.matchingconnect %wire, %reset : !firrtl.inferredreset
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInBlockArgs"
firrtl.circuit "ResetTypeInBlockArgs" {
  // CHECK: firrtl.module @ResetTypeInBlockArgs(in %cond: !firrtl.uint<1>, in %reset: !firrtl.syncreset)
  firrtl.module @ResetTypeInBlockArgs(in %cond: !firrtl.uint<1>, in %reset: !firrtl.inferredreset) {
    // CHECK-NEXT: firrtl.when %cond : !firrtl.uint<1> {
    firrtl.when %cond : !firrtl.uint<1> {
      // CHECK-NEXT: %wire = firrtl.wire : !firrtl.syncreset
      %wire = firrtl.wire : !firrtl.inferredreset
    }
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInInstances"
firrtl.circuit "ResetTypeInInstances" {
  // CHECK: firrtl.module @ResetTypeInInstances(in %reset: !firrtl.syncreset)
  firrtl.module @ResetTypeInInstances(in %reset: !firrtl.inferredreset) {
    // CHECK-NEXT: %child_reset = firrtl.instance child @Child(in reset: !firrtl.syncreset)
    %child_reset = firrtl.instance child @Child(in reset: !firrtl.inferredreset)
    // CHECK-NEXT: firrtl.matchingconnect %child_reset, %reset : !firrtl.syncreset
    firrtl.matchingconnect %child_reset, %reset : !firrtl.inferredreset
  }

  // CHECK: firrtl.module private @Child(in %reset: !firrtl.syncreset)
  firrtl.module private @Child(in %reset: !firrtl.inferredreset) {
  }
}

// CHECK-LABEL: firrtl.circuit "ResetTypeInExtModule"
firrtl.circuit "ResetTypeInExtModule" {
  // CHECK: firrtl.module @ResetTypeInExtModule(in %reset: !firrtl.syncreset)
  firrtl.module @ResetTypeInExtModule(in %reset: !firrtl.inferredreset) {
    // CHECK-NEXT: %ext_reset = firrtl.instance ext @Ext(in reset: !firrtl.syncreset)
    %ext_reset = firrtl.instance ext @Ext(in reset: !firrtl.inferredreset)
    // CHECK-NEXT: firrtl.matchingconnect %ext_reset, %reset : !firrtl.syncreset
    firrtl.matchingconnect %ext_reset, %reset : !firrtl.inferredreset
  }

  // CHECK: firrtl.extmodule @Ext(in reset: !firrtl.syncreset)
  firrtl.extmodule @Ext(in reset: !firrtl.inferredreset)
}

// CHECK-LABEL: firrtl.circuit "MultipleResetTypes"
firrtl.circuit "MultipleResetTypes" {
  // CHECK: firrtl.module @MultipleResetTypes(in %reset1: !firrtl.syncreset, in %reset2: !firrtl.syncreset, out %reset_out: !firrtl.syncreset)
  firrtl.module @MultipleResetTypes(in %reset1: !firrtl.inferredreset, in %reset2: !firrtl.inferredreset, out %reset_out: !firrtl.inferredreset) {
    // CHECK-NEXT: %wire1 = firrtl.wire : !firrtl.syncreset
    %wire1 = firrtl.wire : !firrtl.inferredreset
    // CHECK-NEXT: %wire2 = firrtl.wire : !firrtl.syncreset
    %wire2 = firrtl.wire : !firrtl.inferredreset
    // CHECK-NEXT: firrtl.matchingconnect %wire1, %reset1 : !firrtl.syncreset
    firrtl.matchingconnect %wire1, %reset1 : !firrtl.inferredreset
    // CHECK-NEXT: firrtl.matchingconnect %wire2, %reset2 : !firrtl.syncreset
    firrtl.matchingconnect %wire2, %reset2 : !firrtl.inferredreset
    // CHECK-NEXT: firrtl.matchingconnect %reset_out, %wire1 : !firrtl.syncreset
    firrtl.matchingconnect %reset_out, %wire1 : !firrtl.inferredreset
  }
}

// CHECK-LABEL: firrtl.circuit "RemoveResetAnnotations"
firrtl.circuit "RemoveResetAnnotations" {
  // CHECK: firrtl.module @RemoveResetAnnotations(in %clock: !firrtl.clock, in %reset: !firrtl.syncreset)
  // CHECK-NOT: circt.FullResetAnnotation
  // CHECK-NOT: circt.ExcludeFromFullResetAnnotation
  // CHECK-NOT: sifive.enterprise.firrtl.FullAsyncResetAnnotation
  // CHECK-NOT: sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation
  firrtl.module @RemoveResetAnnotations(
    in %clock: !firrtl.clock,
    in %reset: !firrtl.inferredreset [{class = "circt.FullResetAnnotation", resetType = "async"}]
  ) attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    // CHECK: %wire = firrtl.wire
    // CHECK-NOT: circt.FullResetAnnotation
    %wire = firrtl.wire {annotations = [{class = "sifive.enterprise.firrtl.FullAsyncResetAnnotation"}]} : !firrtl.inferredreset
  }
  // CHECK: firrtl.module private @Child
  // CHECK-NOT: sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation
  firrtl.module private @Child() attributes {annotations = [{class = "sifive.enterprise.firrtl.IgnoreFullAsyncResetAnnotation"}]} {
  }
}
