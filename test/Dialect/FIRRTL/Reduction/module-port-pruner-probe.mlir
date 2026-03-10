// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg true --keep-best=0 --include module-port-pruner | FileCheck %s

// Test removing probe ports that are used by ref.send and ref.define operations.
// This tests the fix for a crash where operations using port arguments were not
// properly erased before removing the ports.
firrtl.circuit "ProbePortRemoval" {
  // CHECK-LABEL: firrtl.module private @Bar
  // CHECK-NOT: in %a
  // CHECK-NOT: out %b
  // CHECK-NOT: out %c
  // CHECK-SAME: {
  firrtl.module private @Bar(
    in %a: !firrtl.uint<65>,
    out %b: !firrtl.probe<uint<65>>,
    out %c: !firrtl.rwprobe<uint<65>>
  ) {
    // CHECK: %wire = firrtl.wire sym @sym
    // CHECK-NOT: firrtl.ref.send
    // CHECK-NOT: firrtl.ref.define
    // CHECK-NOT: firrtl.ref.rwprobe
    %wire = firrtl.wire sym @sym : !firrtl.uint<65>
    %0 = firrtl.ref.send %a : !firrtl.uint<65>
    firrtl.ref.define %b, %0 : !firrtl.probe<uint<65>>
    %1 = firrtl.ref.rwprobe <@Bar::@sym> : !firrtl.rwprobe<uint<65>>
    firrtl.ref.define %c, %1 : !firrtl.rwprobe<uint<65>>
  }
  firrtl.extmodule @ProbePortRemoval()
}

