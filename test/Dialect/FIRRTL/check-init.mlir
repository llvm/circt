// RUN: circt-opt -allow-unregistered-dialect --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file --verify-diagnostics %s | FileCheck %s

// missing wire init
// CHECK: firrtl.circuit "simplewire"
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
  }
}

// -----

// missing output
// CHECK: firrtl.circuit "simpleout"
firrtl.circuit "simpleout"   {
  // expected-error @below {{Port `a` is not initialized.}}
  firrtl.module @simpleout(out %a: !firrtl.uint<1>) {
  }
}


// -----

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

// missing wire init
// CHECK: firrtl.circuit "bundlewire"
firrtl.circuit "bundlewire"   {
  firrtl.module @bundlewire(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized at field `b`.}}
    %x = firrtl.wire  : !firrtl.bundle<a: uint<1>, b:uint<2>>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t, %c : !firrtl.uint<1>
  }
}

// -----

// missing port init
// CHECK: firrtl.circuit "bundleport"
firrtl.circuit "bundleport" {
  // expected-error @below {{Port `x` is not fully initialized at field `b`.}}
  firrtl.module @bundleport(in %a: !firrtl.uint<1>, out %x: !firrtl.bundle<a: uint<1>, b:uint<2>>) {
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t, %a : !firrtl.uint<1>
  }
}

// -----

// missing wire init
// CHECK: firrtl.circuit "bundlewirenest"
firrtl.circuit "bundlewirenest"   {
  firrtl.module @bundlewirenest(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized at field `a.c`.}}
    %x = firrtl.wire  : !firrtl.bundle<a: bundle<b: uint<1>, c:uint<2>>>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    %s = firrtl.subfield %x[a] : !firrtl.bundle<a: bundle<b: uint<1>, c:uint<2>>>
    %t = firrtl.subfield %s[b] : !firrtl.bundle<b: uint<1>, c:uint<2>>
    firrtl.strictconnect %t, %c : !firrtl.uint<1>
  }
}

// -----

// missing port init
// CHECK: firrtl.circuit "bundleportnest"
firrtl.circuit "bundleportnest" {
  // expected-error @below {{Port `x` is not fully initialized at field `a.c`.}}
  firrtl.module @bundleportnest(in %a: !firrtl.uint<1>, out %x: !firrtl.bundle<a: bundle<b: uint<1>, c:uint<2>>>) {
    %s = firrtl.subfield %x[a] : !firrtl.bundle<a: bundle<b: uint<1>, c:uint<2>>>
    %t = firrtl.subfield %s[b] : !firrtl.bundle<b: uint<1>, c:uint<2>>
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

// when wire init
// CHECK: firrtl.circuit "simplewhen"
firrtl.circuit "simplewhen"   {
  firrtl.module @simplewhen(in %c: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.when %c : !firrtl.uint<1> {
      firrtl.strictconnect %x, %c : !firrtl.uint<1>
    } else {
    }
  }
}

// -----

// when wire init
// CHECK: firrtl.circuit "simplewhen"
firrtl.circuit "simplewhen"   {
  firrtl.module @simplewhen(in %c: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
    firrtl.when %c : !firrtl.uint<1> {
    } else {
      firrtl.strictconnect %x, %c : !firrtl.uint<1>
    }
  }
}
