// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-check-init)))' --split-input-file --verify-diagnostics %s

// missing wire init
firrtl.circuit "simplewire"   {
  firrtl.module @simplewire() {
  // expected-error @below {{Wire is not initialized.}}
    %x = firrtl.wire  : !firrtl.uint<1>
  }
}

// -----

// missing output
firrtl.circuit "simpleout"   {
  // expected-error @below {{Port `a` is not initialized.}}
  firrtl.module @simpleout(out %a: !firrtl.uint<1>) {
  }
}


// -----

// missing wire init
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

// missing wire init
firrtl.circuit "bundlewireflip"   {
  firrtl.module @bundlewireflip(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized at field `b`.}}
    %x = firrtl.wire  : !firrtl.bundle<a: uint<1>,  b flip:uint<2>>
    %c = firrtl.constant 1 : !firrtl.uint<1>
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.strictconnect %t, %c : !firrtl.uint<1>
  }
}


// -----

// missing port init
firrtl.circuit "bundleport" {
  // expected-error @below {{Port `x` is not fully initialized at field `b`.}}
  firrtl.module @bundleport(in %a: !firrtl.uint<1>, out %x: !firrtl.bundle<a: uint<1>, b:uint<2>>) {
    %t = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t, %a : !firrtl.uint<1>
  }
}

// -----

// missing wire init
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

// -----

// CHECK: firrtl.circuit "vecSimple"
firrtl.circuit "vecSimple" {
  // expected-error @below {{Port `x` is not fully initialized at field `[1]`.}}
  firrtl.module @vecSimple(in %a: !firrtl.uint<1>, out %x: !firrtl.vector<uint<1>, 2>) {
    %t0 = firrtl.subindex %x[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %t0, %a : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "vecSimpleW"
firrtl.circuit "vecSimpleW" {
  firrtl.module @vecSimpleW(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized at field `[1]`.}}
    %x = firrtl.wire : !firrtl.vector<uint<1>, 2>
    %t0 = firrtl.subindex %x[0] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %t0, %a : !firrtl.uint<1>
  }
}

// -----

// CHECK: firrtl.circuit "vecbun"
firrtl.circuit "vecbun" {
  firrtl.module @vecbun(in %a: !firrtl.uint<1>) {
  // expected-error @below {{Wire is not fully initialized at field `[0].b`.}}
    %x = firrtl.wire : !firrtl.vector<bundle< a: uint<1>, b: uint<2>>, 1>
    %t0 = firrtl.subindex %x[0] : !firrtl.vector<bundle< a: uint<1>, b: uint<2>>, 1>
    %t1 = firrtl.subfield %t0[a] : !firrtl.bundle< a: uint<1>, b: uint<2>>
    firrtl.strictconnect %t1, %a : !firrtl.uint<1>
  }
}


// -----

// CHECK: firrtl.circuit "vecbunbase"
firrtl.circuit "vecbunbase" {
  firrtl.module @vecbunbase(in %a: !firrtl.bundle<a: uint<1>, b: uint<2>>) {
  // expected-error @below {{Wire is not fully initialized at field `[1].a`.}}
  // expected-error @below {{Wire is not fully initialized at field `[1].b`.}}
    %x = firrtl.wire : !firrtl.vector<bundle< a: uint<1>, b: uint<2>>, 2>
    %t0 = firrtl.subindex %x[0] : !firrtl.vector<bundle< a: uint<1>, b: uint<2>>, 2>
    firrtl.strictconnect %t0, %a : !firrtl.bundle<a: uint<1>, b: uint<2> >
  }
}
