// RUN: circt-opt --firrtl-probes-to-signals --split-input-file %s | FileCheck %s --implicit-check-not firrtl.probe


// CHECK-LABEL: "SimpleOneLevel"
firrtl.circuit "SimpleOneLevel" {
  firrtl.module private @Test(out %x: !firrtl.probe<uint<2>>) {
    %w = firrtl.wire : !firrtl.uint<2>
    %zero = firrtl.constant 0 : !firrtl.uint<2>
    firrtl.matchingconnect %w, %zero : !firrtl.uint<2>
    %1 = firrtl.ref.send %w : !firrtl.uint<2>
    firrtl.ref.define %x, %1 : !firrtl.probe<uint<2>>
    // CHECK: matchingconnect %x, %w
  }
  firrtl.module @SimpleOneLevel() {
    // CHECK: %[[X:.+]] = firrtl.instance test
    %test_x = firrtl.instance test @Test(out x: !firrtl.probe<uint<2>>)
    %x = firrtl.ref.resolve %test_x : !firrtl.probe<uint<2>>
    // CHECK: %n = firrtl.node %[[X]]
    %n = firrtl.node %x : !firrtl.uint<2>
  }
}

// -----

// CHECK-LABEL: "RefDefineAndCastWidths"
firrtl.circuit "RefDefineAndCastWidths" {
  // CHECK: @RefDefineAndCastWidths(
  // CHECK-SAME: out %p: !firrtl.uint
  firrtl.module @RefDefineAndCastWidths(in %x: !firrtl.uint<2>, out %p : !firrtl.probe<uint>) {
    // Wire is inserted to handle the behavior of the cast (via firrtl.connect).
    // CHECK-NEXT: %[[W:.+]] = firrtl.wire : !firrtl.uint
    // CHECK-NEXT: firrtl.connect %[[W]], %x
    // CHECK-NEXT: firrtl.connect %p, %[[W]]
    %ref = firrtl.ref.send %x : !firrtl.uint<2>
    %cast = firrtl.ref.cast %ref : (!firrtl.probe<uint<2>>) -> !firrtl.probe<uint>
    firrtl.ref.define %p, %cast : !firrtl.probe<uint>
  }
}

// -----

// Check trickier transform, and handling of ref.send, ref.sub, aliases, and ref.define.

// CHECK-LABEL: "TestP"
firrtl.circuit "TestP" {
  // CHECK: out %p: !firrtl.uint<1>
  firrtl.module @TestP(in %cond: !firrtl.uint<1>, in %d: !firrtl.alias<T, bundle<foo: uint<1>, bar: uint<5>>>, out %p: !firrtl.probe<uint<1>>) {
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.alias<T,
    %w = firrtl.wire : !firrtl.probe<alias<T, bundle<foo: uint<1>, bar: uint<5>>>>
    // ref.sub from below placed early.
    // CHECK-NEXT: %[[SUB:.+]] = firrtl.subfield %w[foo]
    %1 = firrtl.ref.send %d : !firrtl.alias<T, bundle<foo: uint<1>, bar: uint<5>>>
    // CHECK-NEXT: firrtl.when
    // CHECK-NEXT: }
    // CHECK-NEXT: firrtl.matchingconnect %w, %d
    // CHECK-NEXT: firrtl.matchingconnect %p, %[[SUB]]
    firrtl.when %cond : !firrtl.uint<1> {
      firrtl.ref.define %w, %1 : !firrtl.probe<alias<T, bundle<foo: uint<1>, bar: uint<5>>>>

      %0 = firrtl.ref.sub %w[0] : !firrtl.probe<alias<T, bundle<foo: uint<1>, bar: uint<5>>>>
      firrtl.ref.define %p, %0 : !firrtl.probe<uint<1>>
    }
  }
}

// -----

// Extmodule

// CHECK-LABEL: "ExtModule"
firrtl.circuit "ExtModule" {
  // CHECK: out p: !firrtl.alias<T, bundle<foo
  firrtl.extmodule @ExtModule(out p: !firrtl.probe<alias<T, bundle<foo: uint<1>, bar: uint<5>>>>)
}

// -----

// CHIRRTL debug port

// CHECK-LABEL: "DbgsMemPort"
firrtl.circuit "DbgsMemPort" {
  firrtl.module @DbgsMemPort(in %clock: !firrtl.clock, in %addr : !firrtl.uint<1>, out %_a: !firrtl.probe<vector<uint<1>, 2>>, in %cond : !firrtl.uint<1>) {
    %ram = chirrtl.combmem : !chirrtl.cmemory<uint<1>, 2>
    // CHECK: %[[W:.+]] = firrtl.wire : !firrtl.vector<uint<1>, 2>

    // Not put under when usually, but for testing ensure handle.
    firrtl.when %cond : !firrtl.uint<1> {
      %port0_data = chirrtl.debugport %ram {name = "port0"} : (!chirrtl.cmemory<uint<1>, 2>) -> !firrtl.probe<vector<uint<1>, 2>>
      firrtl.ref.define %_a, %port0_data : !firrtl.probe<vector<uint<1>, 2>>
    }
    // Inserted ports must go after existing one.
    // CHECK: "ramport"
    %ramport_data, %ramport_port = chirrtl.memoryport Read %ram {name = "ramport"} : (!chirrtl.cmemory<uint<1>, 2>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
  
    firrtl.when %cond : !firrtl.uint<1> {
      chirrtl.memoryport.access %ramport_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint<1>, !firrtl.clock
    }

    // CHECK: %[[tap_0_data:.+]], %[[tap_0_port:.+]] = chirrtl.memoryport Read %ram {name = "memTap_0"} : (!chirrtl.cmemory<uint<1>, 2>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
    // CHECK: %[[ZERO:.+]] = firrtl.constant 0
    // CHECK: memoryport.access %[[tap_0_port]][%[[ZERO]]]
    // CHECK: %[[tap_1_data:.+]], %[[tap_1_port:.+]] = chirrtl.memoryport Read %ram {name = "memTap_1"} : (!chirrtl.cmemory<uint<1>, 2>) -> (!firrtl.uint<1>, !chirrtl.cmemoryport)
    // CHECK: %[[ONE:.+]] = firrtl.constant 1
    // CHECK: memoryport.access %[[tap_1_port]][%[[ONE]]]

    // CHECK: %[[DATA:.+]] = firrtl.vectorcreate %[[tap_0_data]], %[[tap_1_data]]

    // CHECK: matchingconnect %[[W]], %[[DATA]]
    // CHECK: matchingconnect %_a, %[[W]]
  }
}
