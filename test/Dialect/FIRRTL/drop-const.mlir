// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-drop-const))' %s | FileCheck %s --implicit-check-not=const.
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
// CHECK-SAME: out g: !firrtl.probe<uint<1>>)
firrtl.extmodule @ConstPortExtModule(
  in a: !firrtl.const.uint<1>, 
  in b: !firrtl.const.bundle<a: uint<1>>,
  in c: !firrtl.bundle<a: const.uint<1>>,
  in d: !firrtl.const.vector<uint<1>, 3>,
  in e: !firrtl.vector<const.uint<1>, 3>,
  in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  out g: !firrtl.probe<const.uint<1>>
)

// Const is dropped from module signature and ops
// CHECK-LABEL: firrtl.module @ConstPortModule(
// CHECK-SAME: in %a: !firrtl.uint<1>
// CHECK-SAME: in %b: !firrtl.bundle<a: uint<1>>
// CHECK-SAME: in %c: !firrtl.bundle<a: uint<1>>,
// CHECK-SAME: in %d: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %e: !firrtl.vector<uint<1>, 3>,
// CHECK-SAME: in %f: !firrtl.enum<a: uint<2>, b: uint<1>>,
// CHECK-SAME: out %g: !firrtl.probe<uint<1>>)
firrtl.module @ConstPortModule(
  in %a: !firrtl.const.uint<1>, 
  in %b: !firrtl.const.bundle<a: uint<1>>,
  in %c: !firrtl.bundle<a: const.uint<1>>,
  in %d: !firrtl.const.vector<uint<1>, 3>,
  in %e: !firrtl.vector<const.uint<1>, 3>,
  in %f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
  out %g: !firrtl.probe<const.uint<1>>
) {
  // CHECK-NEXT: firrtl.instance inst @ConstPortExtModule(
  // CHECK-SAME: in a: !firrtl.uint<1>
  // CHECK-SAME: in b: !firrtl.bundle<a: uint<1>>
  // CHECK-SAME: in c: !firrtl.bundle<a: uint<1>>,
  // CHECK-SAME: in d: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in e: !firrtl.vector<uint<1>, 3>,
  // CHECK-SAME: in f: !firrtl.enum<a: uint<2>, b: uint<1>>,
  // CHECK-SAME: out g: !firrtl.probe<uint<1>>)
  %a2, %b2, %c2, %d2, %e2, %f2, %g2 = firrtl.instance inst @ConstPortExtModule(
    in a: !firrtl.const.uint<1>, 
    in b: !firrtl.const.bundle<a: uint<1>>,
    in c: !firrtl.bundle<a: const.uint<1>>,
    in d: !firrtl.const.vector<uint<1>, 3>,
    in e: !firrtl.vector<const.uint<1>, 3>,
    in f: !firrtl.const.enum<a: uint<2>, b: uint<1>>,
    out g: !firrtl.probe<const.uint<1>>
  )
  %a2_read, %a2_write = firrtl.deduplex %a2 : !firrtl.const.uint<1>
  firrtl.strictconnect %a2_write, %a : !firrtl.const.uint<1>
  %b2_read, %b2_write = firrtl.deduplex %b2 : !firrtl.const.bundle<a: uint<1>>
  firrtl.strictconnect %b2_write, %b : !firrtl.const.bundle<a: uint<1>>
  %c2_read, %c2_write = firrtl.deduplex %c2 : !firrtl.bundle<a: const.uint<1>>
  firrtl.strictconnect %c2_write, %c : !firrtl.bundle<a: const.uint<1>>
  %d2_read, %d2_write = firrtl.deduplex %d2 : !firrtl.const.vector<uint<1>, 3>
  firrtl.strictconnect %d2_write, %d : !firrtl.const.vector<uint<1>, 3>
  %e2_read, %e2_write = firrtl.deduplex %e2 : !firrtl.vector<const.uint<1>,3>
  firrtl.strictconnect %e2_write, %e : !firrtl.vector<const.uint<1>, 3>
  %f2_read, %f2_write = firrtl.deduplex %f2 : !firrtl.const.enum<a: uint<2>, b: uint<1>>
  firrtl.strictconnect %f2_write, %f : !firrtl.const.enum<a: uint<2>, b: uint<1>>
  firrtl.ref.define %g, %g2 : !firrtl.probe<const.uint<1>>
}

// Const-cast ops are erased
// CHECK-LABEL: firrtl.module @ConstCastErase
firrtl.module @ConstCastErase(in %in: !firrtl.const.uint<1>, out %out: !firrtl.uint<1>) {
  // CHECK-NOT: firrtl.constCast
  // CHECK-NEXT: firrtl.strictconnect %out, %in : !firrtl.uint<1>
  %0 = firrtl.constCast %in : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
  %out_read, %out_write = firrtl.deduplex %out : !firrtl.uint<1>
  firrtl.strictconnect %out_write, %0 : !firrtl.uint<1> 
}

// Const is dropped within when blocks
// CHECK-LABEL: firrtl.module @ConstDropInWhenBlock
firrtl.module @ConstDropInWhenBlock(in %cond: !firrtl.const.uint<1>, in %in1: !firrtl.const.sint<2>, in %in2: !firrtl.const.sint<2>, out %out: !firrtl.const.sint<2>) {
  %out_read, %out_write = firrtl.deduplex %out : !firrtl.const.sint<2>
  // CHECK: firrtl.when %cond : !firrtl.uint<1>
  firrtl.when %cond : !firrtl.const.uint<1> {
    // CHECK: firrtl.strictconnect %out, %in1 : !firrtl.sint<2>
    firrtl.strictconnect %out_write, %in1 : !firrtl.const.sint<2>
  } else {
    // CHECK: firrtl.strictconnect %out, %in2 : !firrtl.sint<2>
    firrtl.strictconnect %out_write, %in2 : !firrtl.const.sint<2>
  }
}
}
