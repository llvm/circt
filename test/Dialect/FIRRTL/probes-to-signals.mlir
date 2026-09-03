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

// A same-type cast drops forceability without creating a copy wire.

// CHECK-LABEL: "ReadOnlyCastOfRWProbe"
firrtl.circuit "ReadOnlyCastOfRWProbe" {
  // CHECK: @ReadOnlyCastOfRWProbe(
  // CHECK-SAME: out %p: !firrtl.uint<8>
  firrtl.module @ReadOnlyCastOfRWProbe(in %x: !firrtl.uint<8>, out %p: !firrtl.probe<uint<8>>) {
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.uint<8>
    // CHECK-NEXT: firrtl.matchingconnect %w, %x
    // CHECK: firrtl.matchingconnect %p, %w
    // CHECK-NEXT: }
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %w, %x : !firrtl.uint<8>
    %cast = firrtl.ref.cast %w_ref : (!firrtl.rwprobe<uint<8>>) -> !firrtl.probe<uint<8>>
    firrtl.ref.define %p, %cast : !firrtl.probe<uint<8>>
  }
}

// -----

// Check ref.send, ref.sub, aliases, and ref.define.

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

// Extmodule using alias

// CHECK-LABEL: "ExtModule"
firrtl.circuit "ExtModule" {
  // CHECK: out p: !firrtl.alias<T, bundle<foo
  firrtl.extmodule @ExtModule(out p: !firrtl.probe<alias<T, bundle<foo: uint<1>, bar: uint<5>>>>)
}

// -----

// Extmodule RWProbe lowering adds a force-control port.

// CHECK-LABEL: "ExtModuleRW"
firrtl.circuit "ExtModuleRW" {
  // CHECK: out ro: !firrtl.uint<1>
  // CHECK-SAME: out rw: !firrtl.uint<2>
  // CHECK-SAME: in rw_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<2>, clk: clock>
  firrtl.extmodule @ExtModuleRW(out ro: !firrtl.probe<uint<1>>, out rw: !firrtl.rwprobe<uint<2>>)
}

// -----

// An extmodule RWProbe's force-control port propagates to its instance.

// CHECK-LABEL: "ExtModuleRWRead"
firrtl.circuit "ExtModuleRWRead" {
  // CHECK: firrtl.extmodule @ExtRW(out rw: !firrtl.uint<2>, in rw_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<2>, clk: clock>)
  firrtl.extmodule @ExtRW(out rw: !firrtl.rwprobe<uint<2>>)
  // CHECK-LABEL: firrtl.module @ExtModuleRWRead
  firrtl.module @ExtModuleRWRead(out %o: !firrtl.uint<2>) {
    // CHECK: %[[EXT:.+]], %{{.+}} = firrtl.instance e @ExtRW(out rw: !firrtl.uint<2>, in rw_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<2>, clk: clock>)
    %e_rw = firrtl.instance e @ExtRW(out rw: !firrtl.rwprobe<uint<2>>)
    %r = firrtl.ref.resolve %e_rw : !firrtl.rwprobe<uint<2>>
    // CHECK-NEXT: firrtl.matchingconnect %o, %[[EXT]]
    firrtl.matchingconnect %o, %r : !firrtl.uint<2>
  }
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

// -----
// Exported RWProbe lowering adds an appended force-control input.

// CHECK-LABEL: "ForceableRWProbeExport"
firrtl.circuit "ForceableRWProbeExport" {
  // CHECK: @ForceableRWProbeExport(out %p: !firrtl.uint<2>, in %p_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<2>, clk: clock>)
  firrtl.module @ForceableRWProbeExport(out %p : !firrtl.rwprobe<uint<2>>) {
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.uint<2>
    // CHECK-NEXT: %w_forced = firrtl.wire : !firrtl.uint<2>
    // CHECK-NEXT: firrtl.matchingconnect %p, %w_forced
    %w, %w_f = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %w_f : !firrtl.rwprobe<uint<2>>
  }
}

// -----

// A force-control port is renamed when its generated name collides.
// CHECK-LABEL: "ForceControlPortNameCollision"
firrtl.circuit "ForceControlPortNameCollision" {
  // CHECK: firrtl.module private @Child(in %p_force_ctrl: !firrtl.uint<1>, out %p: !firrtl.uint<8>, in %p_force_ctrl_0: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<8>, clk: clock>)
  firrtl.module private @Child(in %p_force_ctrl: !firrtl.uint<1>, out %p: !firrtl.rwprobe<uint<8>>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<8>, !firrtl.rwprobe<uint<8>>
    firrtl.ref.define %p, %w_ref : !firrtl.rwprobe<uint<8>>
  }
  firrtl.module @ForceControlPortNameCollision(in %clock: !firrtl.clock,
                                                in %en: !firrtl.uint<1>,
                                                in %value: !firrtl.uint<8>,
                                                out %o: !firrtl.uint<8>) {
    %i_dummy, %i_p = firrtl.instance i @Child(in p_force_ctrl: !firrtl.uint<1>, out p: !firrtl.rwprobe<uint<8>>)
    %zero = firrtl.constant 0 : !firrtl.uint<1>
    firrtl.matchingconnect %i_dummy, %zero : !firrtl.uint<1>
    firrtl.ref.force %clock, %en, %i_p, %value : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<8>>, !firrtl.uint<8>
    %r = firrtl.ref.resolve %i_p : !firrtl.rwprobe<uint<8>>
    firrtl.matchingconnect %o, %r : !firrtl.uint<8>
  }
}

// -----

// Check reading a forceable RWProbe.

// CHECK-LABEL: "ForceableToRead"
firrtl.circuit "ForceableToRead" {
  // CHECK: @ForceableToRead(
  firrtl.module @ForceableToRead(out %r : !firrtl.uint<2>) {
    // CHECK-NEXT: %w = firrtl.wire : !firrtl.uint<2>
    // CHECK-NEXT: firrtl.matchingconnect %r, %w
    %w, %w_f = firrtl.wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %data = firrtl.ref.resolve %w_f : !firrtl.rwprobe<uint<2>>
    firrtl.matchingconnect %r, %data : !firrtl.uint<2>
  }
}

// -----

// Check RWProbe lowering and its force-control port.

// CHECK-LABEL: "RWProbeOp"
firrtl.circuit "RWProbeOp" {
  // CHECK: @RWProbeOp(out %p: !firrtl.uint<2>, in %p_force_ctrl: !firrtl.bundle<forceActive: uint<1>, releaseActive: uint<1>, forcedValue: uint<2>, clk: clock>)
  firrtl.module @RWProbeOp(out %p: !firrtl.rwprobe<uint<2>>) {
    // CHECK-NEXT: %w = firrtl.wire sym @sym : !firrtl.uint<2>
    // CHECK-NEXT: %w_forced = firrtl.wire : !firrtl.uint<2>
    // CHECK-NEXT: firrtl.matchingconnect %p, %w_forced
    %w = firrtl.wire sym @sym : !firrtl.uint<2>
    %rwprobe = firrtl.ref.rwprobe <@RWProbeOp::@sym> : !firrtl.rwprobe<uint<2>>
    firrtl.ref.define %p, %rwprobe : !firrtl.rwprobe<uint<2>>
  }
}
