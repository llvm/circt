// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(any(firrtl-drop-const)))' %s | FileCheck %s --implicit-check-not=const.
firrtl.circuit "DropConst" {
firrtl.module @DropConst() {}

// Const is dropped from extmodule signature
// CHECK-LABEL: firrtl.extmodule @ConstPortExtModule(
// CHECK-SAME: in a: !firrtl.uint<1>
// CHECK-SAME: in b: !firrtl.bundle<a: uint<1>>
// CHECK-SAME: in c: !firrtl.bundle<a: uint<1>>,
// CHECK-SAME: in d: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in e: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in f: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: in g: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: out h: !firrtl.probe<uint<1>>)
firrtl.extmodule @ConstPortExtModule(
  in a: !firrtl.const.uint<1>, 
  in b: !firrtl.const.bundle<a: uint<1>>,
  in c: !firrtl.bundle<a: const.uint<1>>,
  in d: !firrtl.const.vector<uint<1>, 3>,
  in e: !firrtl.vector<const.uint<1>, 3>,
  in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  in g: !firrtl.enum<a: uint<2>, b: const.uint<1>>,
  out h: !firrtl.probe<const.uint<1>>
)

// Const is dropped from module signature and ops
// CHECK-LABEL: firrtl.module @ConstPortModule(
// CHECK-SAME: in %a: !firrtl.uint<1>
// CHECK-SAME: in %b: !firrtl.bundle<a: uint<1>>
// CHECK-SAME: in %c: !firrtl.bundle<a: uint<1>>,
// CHECK-SAME: in %d: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %e: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %f: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: in %g: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: out %h: !firrtl.probe<uint<1>>)
firrtl.module @ConstPortModule(
  in %a: !firrtl.const.uint<1>, 
  in %b: !firrtl.const.bundle<a: uint<1>>,
  in %c: !firrtl.bundle<a: const.uint<1>>,
  in %d: !firrtl.const.vector<uint<1>, 3>,
  in %e: !firrtl.vector<const.uint<1>, 3>,
  in %f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  in %g: !firrtl.enum<a: uint<2>, b: const.uint<1>>,
  out %h: !firrtl.probe<const.uint<1>>
) {
  // CHECK-NEXT: firrtl.instance inst @ConstPortExtModule(
  // CHECK-SAME: in a: !firrtl.uint<1>
  // CHECK-SAME: in b: !firrtl.bundle<a: uint<1>>
  // CHECK-SAME: in c: !firrtl.bundle<a: uint<1>>,
  // CHECK-SAME: in d: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in e: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in f: !firrtl.enum<a: uint<2>, b: uint<1>>,
  // CHECK-SAME: in g: !firrtl.enum<a: uint<2>, b: uint<1>>,
  // CHECK-SAME: out h: !firrtl.probe<uint<1>>)
  %a2, %b2, %c2, %d2, %e2, %f2, %g2, %h2 = firrtl.instance inst @ConstPortExtModule(
    in a: !firrtl.const.uint<1>, 
    in b: !firrtl.const.bundle<a: uint<1>>,
    in c: !firrtl.bundle<a: const.uint<1>>,
    in d: !firrtl.const.vector<uint<1>, 3>,
    in e: !firrtl.vector<const.uint<1>, 3>,
    in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
    in g: !firrtl.enum<a: uint<2>, b: const.uint<1>>,
    out h: !firrtl.probe<const.uint<1>>
  )

  firrtl.strictconnect %a2, %a : !firrtl.const.uint<1>
  firrtl.strictconnect %b2, %b : !firrtl.const.bundle<a: uint<1>>
  firrtl.strictconnect %c2, %c : !firrtl.bundle<a: const.uint<1>>
  firrtl.strictconnect %d2, %d : !firrtl.const.vector<uint<1>, 3>
  firrtl.strictconnect %e2, %e : !firrtl.vector<const.uint<1>, 3>
  firrtl.strictconnect %f2, %f : !firrtl.const.enum<a: uint<2>, b: uint<1>>
  firrtl.strictconnect %g2, %g : !firrtl.enum<a: uint<2>, b: const.uint<1>>
  firrtl.ref.define %h, %h2 : !firrtl.probe<const.uint<1>>
}

// Const-cast ops are erased
// CHECK-LABEL: firrtl.module @ConstCastErase
firrtl.module @ConstCastErase(in %in: !firrtl.const.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK-NOT: firrtl.constCast
  // CHECK-NEXT: firrtl.strictconnect %out, %in : !firrtl.uint<1>
  %0 = firrtl.constCast %in : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
  firrtl.strictconnect %out, %0 : !firrtl.uint<1> 
}

// Const connections can occur within const-conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionConstAssign
firrtl.module @ConstConditionConstAssign(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.const.sint<2>, in %in2: !firrtl.const.sint<2>, out %out: !firrtl.const.sint<2>) {
  // CHECK: firrtl.when %cond : !firrtl.uint<1>
  firrtl.when %cond : !firrtl.const.uint<1> {
    // CHECK: firrtl.strictconnect %out, %in1 : !firrtl.sint<2>
    firrtl.strictconnect %out, %in1 : !firrtl.const.sint<2>
  } else {
    // CHECK: firrtl.strictconnect %out, %in2 : !firrtl.sint<2>
    firrtl.strictconnect %out, %in2 : !firrtl.const.sint<2>
  }
}

// Non-const connections can occur within const-conditioned whens
// CHECK-LABEL: firrtl.module @ConstConditionNonConstAssign
firrtl.module @ConstConditionNonConstAssign(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.sint<2>, in %in2: !firrtl.sint<2>, out %out: !firrtl.sint<2>) {
  // CHECK: firrtl.when %cond : !firrtl.uint<1>
  firrtl.when %cond : !firrtl.const.uint<1> {
    firrtl.strictconnect %out, %in1 : !firrtl.sint<2>
  } else {
    firrtl.strictconnect %out, %in2 : !firrtl.sint<2>
  }
}

// Const connections can occur when the destination is local to a non-const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstAssign
firrtl.module @NonConstWhenLocalConstAssign(in %cond: !firrtl.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    // CHECK:      firrtl.wire : !firrtl.uint<9>
    // CHECK-NEXT: firrtl.constant 0 : !firrtl.uint<9>
    %w = firrtl.wire : !firrtl.const.uint<9>
    %c = firrtl.constant 0 : !firrtl.const.uint<9>
    firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
  }
}

// Const connections can occur when the destination is local to a non-const 
// conditioned when block and the connection is inside a const conditioned when block
// CHECK-LABEL: firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign
firrtl.module @NonConstWhenLocalConstNestedConstWhenAssign(in %cond: !firrtl.uint<1>, in %constCond: !firrtl.const.uint<1>) {
  firrtl.when %cond : !firrtl.uint<1> {
    // CHECK: firrtl.wire : !firrtl.uint<9>
    %w = firrtl.wire : !firrtl.const.uint<9>
    // CHECK-NEXT: firrtl.when %constCond : !firrtl.uint<1>
    firrtl.when %constCond : !firrtl.const.uint<1> {
      %c = firrtl.constant 0 : !firrtl.const.uint<9>
      firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
    } else {
      %c = firrtl.constant 1 : !firrtl.const.uint<9>
      firrtl.strictconnect %w, %c : !firrtl.const.uint<9>
    }
  }
}

firrtl.module @NonConstWhenConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.const.bundle<a flip: uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.const.bundle<a flip: uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}

firrtl.module @NonConstWhenNestedConstFlipAssign(in %p: !firrtl.uint<1>, in %in: !firrtl.bundle<a flip: uint<2>>, out %out: !firrtl.bundle<a flip: const.uint<2>>) {
  firrtl.when %p : !firrtl.uint<1> {
    firrtl.connect %out, %in : !firrtl.bundle<a flip: const.uint<2>>, !firrtl.bundle<a flip: uint<2>>
  }
}
}