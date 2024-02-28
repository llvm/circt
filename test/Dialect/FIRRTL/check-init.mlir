// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file %s | FileCheck %s
// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file --verify-diagnostics %s | FileCheck %s


// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
    %x = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %x, %c : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "simpleout"
firrtl.circuit "simpleout"   {
  firrtl.module @simpleout(out %a: !firrtl.uint<1>) {
    %c = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %a, %c : !firrtl.uint<1>
  }
}

// -----

// non missing port init due to flip
// CHECK: firrtl.circuit "bundleportflip"
firrtl.circuit "bundleportflip" {
  firrtl.module @bundleportflip(in %a: !firrtl.uint<1>, out %x: !firrtl.bundle<a: uint<1>, b flip:uint<2>>) {
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.strictconnect %t, %a : !firrtl.uint<1>
  }
}

// -----

// when wire init
// CHECK: firrtl.circuit "simplewhen"
firrtl.circuit "simplewhen"   {
  firrtl.module @simplewhen(in %c: !firrtl.uint<1>) {
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.when %c : !firrtl.uint<1> {
      firrtl.strictconnect %x, %c : !firrtl.uint<1>
    } else {
      firrtl.strictconnect %x, %c : !firrtl.uint<1>
    }
  }
}

// -----

// CHECK: firrtl.circuit "simpleoutref"
firrtl.circuit "simpleoutref"   {
  firrtl.module @simpleoutref(out %a: !firrtl.probe<uint<1>>) {
    %x = firrtl.wire  : !firrtl.uint<1>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.strictconnect %x, %c : !firrtl.uint<1>
    %r = firrtl.ref.send %x : !firrtl.uint<1>
    firrtl.ref.define %a, %r : !firrtl.probe<uint<1>>
  }
}

