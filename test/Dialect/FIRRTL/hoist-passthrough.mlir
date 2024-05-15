// Check transform output.
// RUN: circt-opt %s -split-input-file --firrtl-hoist-passthrough | FileCheck %s

// Simple example: HW passthrough.

// CHECK-LABEL: "SimpleHW"
firrtl.circuit "SimpleHW" {
  // CHECK:      module private @UTurn(in %in: !firrtl.uint<1>) {
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  firrtl.module @SimpleHW(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    // CHECK: %[[HW_IN:.+]] = firrtl.instance
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    // CHECK: strictconnect %out, %[[HW_IN]]
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Multiple outputs derived from same input.

// CHECK-LABEL: "Split"
firrtl.circuit "Split" {
  // CHECK: @SPassthrough(in %in: !firrtl.bundle<a: uint<5>, b: uint<5>>) {
  firrtl.module private @SPassthrough(in %in: !firrtl.bundle<a: uint<5>, b: uint<5>>, out %y: !firrtl.uint<5>, out %z: !firrtl.uint<5>) {
    // %w = firrtl.wire : !firrtl.bundle<a: uint<5>, b: uint<5>>
    // firrtl.strictconnect %w, %in : !firrtl.bundle<a: uint<5>, b: uint<5>>

    %0 = firrtl.subfield %in[b] : !firrtl.bundle<a: uint<5>, b: uint<5>>
    %1 = firrtl.subfield %in[a] : !firrtl.bundle<a: uint<5>, b: uint<5>>
    firrtl.strictconnect %y, %1 : !firrtl.uint<5>
    firrtl.strictconnect %z, %0 : !firrtl.uint<5>
  }
  firrtl.module @Split(in %x: !firrtl.bundle<a: uint<5>, b: uint<5>>, out %y: !firrtl.uint<5>, out %z: !firrtl.uint<5>) {
    %sp_in, %sp_y, %sp_z = firrtl.instance sp interesting_name @SPassthrough(in in: !firrtl.bundle<a: uint<5>, b: uint<5>>, out y: !firrtl.uint<5>, out z: !firrtl.uint<5>)
    firrtl.strictconnect %sp_in, %x : !firrtl.bundle<a: uint<5>, b: uint<5>>
    firrtl.strictconnect %y, %sp_y : !firrtl.uint<5>
    firrtl.strictconnect %z, %sp_z : !firrtl.uint<5>
  }
}

// -----

// Multiple-level hoist

// CHECK-LABEL: HWMultiLevel
firrtl.circuit "HWMultiLevel" {
  // CHECK:      @Child(in %x: !firrtl.uint<5>) {
  // CHECK-NEXT: }
  firrtl.module private @Child(in %x: !firrtl.uint<5>, out %y: !firrtl.uint<5>) {
    firrtl.strictconnect %y, %x : !firrtl.uint<5>
  }
  // CHECK:      @Mid(in %x: !firrtl.uint<5>) {
  // CHECK-NEXT:   %[[C_IN:.+]] = firrtl.instance c
  // CHECK-NEXT:   firrtl.strictconnect %[[C_IN]], %x
  // CHECK-NEXT: }
  firrtl.module private @Mid(in %x: !firrtl.uint<5>, out %y: !firrtl.uint<5>) {
    %c_x, %c_y = firrtl.instance c interesting_name @Child(in x: !firrtl.uint<5>, out y: !firrtl.uint<5>)
    firrtl.strictconnect %c_x, %x : !firrtl.uint<5>
    firrtl.strictconnect %y, %c_y : !firrtl.uint<5>
  }
  // CHECK: @HWMultiLevel
  firrtl.module @HWMultiLevel(in %x: !firrtl.uint<5>, in %x2: !firrtl.uint<5>, out %y: !firrtl.uint<5>) {
    // CHECK: %[[M_X:.+]] = firrtl.instance m
    // CHECK: %[[M2_X:.+]] = firrtl.instance m2
    %m_x, %m_y = firrtl.instance m interesting_name @Mid(in x: !firrtl.uint<5>, out y: !firrtl.uint<5>)
    %m2_x, %m2_y = firrtl.instance m2 interesting_name @Mid(in x: !firrtl.uint<5>, out y: !firrtl.uint<5>)
    firrtl.strictconnect %m_x, %x : !firrtl.uint<5>
    firrtl.strictconnect %m2_x, %x2 : !firrtl.uint<5>

    // CHECK: firrtl.and %[[M_X]], %[[M2_X]]
    %0 = firrtl.and %m_y, %m2_y : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
    firrtl.strictconnect %y, %0 : !firrtl.uint<5>
  }
}

// -----

// Don't hoist through public modules

// CHECK-LABEL: "Public"
firrtl.circuit "Public" {
  // CHECK: @Public(in %in
  // CHECK-SAME: , out %out
  firrtl.module @Public(in %in : !firrtl.uint<5>, out %out : !firrtl.uint<5>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<5>
  }
}

// -----

// Check insertion of temporary.
// CHECK-LABEL: "NeedsWire"
firrtl.circuit "NeedsWire" {
  firrtl.module @NeedsWire(in %in_cond : !firrtl.vector<uint<1>, 2>) {
    %c_cond, %c_out = firrtl.instance c @Child(in cond : !firrtl.vector<uint<1>, 2>,
                                               out out : !firrtl.uint<1>)
    firrtl.strictconnect %c_cond, %in_cond : !firrtl.vector<uint<1>, 2>
  }
  firrtl.extmodule @ExtSink(in sink : !firrtl.uint<1>)
  // CHECK: module private @Child(
  // CHECK-NOT: out
  firrtl.module private @Child(in %cond : !firrtl.vector<uint<1>, 2>,
                               out %out : !firrtl.uint<1>) {
    // CHECK-NEXT: firrtl.wire : !firrtl.uint<1>
    %sub = firrtl.subindex %cond[1] : !firrtl.vector<uint<1>, 2>
    firrtl.strictconnect %out, %sub : !firrtl.uint<1>

    %sink = firrtl.instance ev @ExtSink(in sink : !firrtl.uint<1>)
    firrtl.strictconnect %sink, %out : !firrtl.uint<1>
  }
}

// -----

// Reject through cast: hw

// Some casts are reasonable to handle, future work.

// CHECK-LABEL: "HWCast"
firrtl.circuit "HWCast" {
  // CHECK:      module private @UTurnHWCast(in %in: !firrtl.const.uint<5>, out %out: !firrtl.uint<5>) {
  firrtl.module private @UTurnHWCast(in %in : !firrtl.const.uint<5>, out %out : !firrtl.uint<5>) {
    %cast = firrtl.constCast %in : (!firrtl.const.uint<5>) -> !firrtl.uint<5>
    firrtl.strictconnect %out, %cast : !firrtl.uint<5>
  }
  firrtl.module @HWCast(in %in : !firrtl.const.uint<5>, out %out : !firrtl.uint<5>) {
    // CHECK: %{{.+}}, %[[U_OUT:.+]] = firrtl.instance
    %u_in, %u_out = firrtl.instance u @UTurnHWCast(in in : !firrtl.const.uint<5>, out out : !firrtl.uint<5>)
    firrtl.strictconnect %u_in, %in : !firrtl.const.uint<5>
    // CHECK: firrtl.strictconnect %out, %[[U_OUT]] : !firrtl.uint<5>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<5>
  }
}

// -----

// Hoisting through wires: hw

// CHECK-LABEL: HWWire
firrtl.circuit "HWWire" {
  // CHECK:      module private @UWire(in %in: !firrtl.uint<1>) {
  // CHECK-NEXT:   %[[W:.+]] = firrtl.wire
  // CHECK-NEXT:   firrtl.strictconnect %[[W]], %in
  // CHECK-NEXT: }
  firrtl.module private @UWire(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module @HWWire(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    // CHECK: %[[HW_IN:.+]] = firrtl.instance
    %u_in, %u_out = firrtl.instance u @UWire(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    // CHECK: strictconnect %out, %[[HW_IN]]
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Hoisting reset signal: HW

// CHECK-LABEL: HWReset
firrtl.circuit "HWReset" {
  // CHECK:      module private @UTurn(in %in: !firrtl.reset) {
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.reset,
                               out %out : !firrtl.reset) {
    firrtl.strictconnect %out, %in : !firrtl.reset
  }
  firrtl.module @HWReset(in %in : !firrtl.asyncreset, out %out : !firrtl.reset) {
    // CHECK: %[[HW_IN:.+]] = firrtl.instance
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.reset,
                                             out out : !firrtl.reset)
    // CHECK: strictconnect %out, %[[HW_IN]]
    firrtl.connect %u_in, %in : !firrtl.reset, !firrtl.asyncreset
    firrtl.strictconnect %out, %u_out : !firrtl.reset
  }
}

// -----

// Don't infinite loop on cycles

// CHECK-LABEL: "SimpleCycle"
firrtl.circuit "SimpleCycle" {
  firrtl.module private @UTurn(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    // Connectivity cycle.
    %w = firrtl.wire : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
    firrtl.strictconnect %w, %out : !firrtl.uint<1>
  }
  firrtl.module @SimpleCycle(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Reject if symbol on source.

// CHECK-LABEL: "SymSource"
firrtl.circuit "SymSource" {
  // CHECK:      module private @UTurn(in %in: !firrtl.uint<1> sym @sym, out %out
  // CHECK-NEXT:   firrtl.strictconnect
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.uint<1> sym @sym,
                               out %out : !firrtl.uint<1>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  firrtl.module @SymSource(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Reject if symbol on dest.

// CHECK-LABEL: "SymDest"
firrtl.circuit "SymDest" {
  // CHECK:      module private @UTurn(in %in: !firrtl.uint<1>, out %out
  // CHECK-NEXT:   firrtl.strictconnect
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1> sym @sym) {
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  firrtl.module @SymDest(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Reject if symbol on intermediate.

// CHECK-LABEL: SymIntermediate
firrtl.circuit "SymIntermediate" {
  // CHECK:      module private @UWire(in %in: !firrtl.uint<1>, out
  // CHECK-NEXT:   %[[W:.+]] = firrtl.wire sym @sym
  firrtl.module private @UWire(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %w = firrtl.wire sym @sym : !firrtl.uint<1>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module @SymIntermediate(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UWire(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Symbol on instance means keep symbol, not dontTouch its ports.

// CHECK-LABEL: SymInstance
firrtl.circuit "SymInstance" {
  firrtl.module private @Sink(in %in: !firrtl.uint<1>) {}
  // CHECK: @UWire(in %in: !firrtl.uint<1>)
  firrtl.module private @UWire(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %sink_in = firrtl.instance sink sym @blocker @Sink(in in: !firrtl.uint<1>)
    firrtl.strictconnect %sink_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %sink_in : !firrtl.uint<1>
  }
  firrtl.module @SymInstance(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UWire(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Through instance input port.

// CHECK-LABEL: "Instance"
firrtl.circuit "Instance" {
  firrtl.module private @Sink(in %in: !firrtl.uint<1>) {}
  // CHECK: @UWire(in %in: !firrtl.uint<1>) {
  firrtl.module private @UWire(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %sink_in = firrtl.instance sink @Sink(in in: !firrtl.uint<1>)
    firrtl.strictconnect %sink_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %sink_in : !firrtl.uint<1>
  }
  firrtl.module @Instance(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UWire(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Reject non-passive source.

// CHECK-LABEL: "NonPassiveSource"
firrtl.circuit "NonPassiveSource" {
  // CHECK: @Child
  // CHECK-SAME: flip
  // CHECK-SAME: out %y
  firrtl.module private @Child(in %x: !firrtl.bundle<a: uint<1>, b flip: uint<2>>, out %y: !firrtl.uint<1>) {
    %0 = firrtl.subfield %x[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %1 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.strictconnect %y, %1 : !firrtl.uint<1>
    %c1_ui2 = firrtl.constant 1 : !firrtl.uint<2>
    firrtl.strictconnect %0, %c1_ui2 : !firrtl.uint<2>
  }
  firrtl.module @NonPassiveSource(in %x: !firrtl.bundle<a: uint<1>, b flip: uint<2>>, out %y: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
    %c_x, %c_y = firrtl.instance c @Child(in x: !firrtl.bundle<a: uint<1>, b flip: uint<2>>, out y: !firrtl.uint<1>)
    %0 = firrtl.subfield %c_x[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %1 = firrtl.subfield %x[a] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.strictconnect %0, %1 : !firrtl.uint<1>
    %2 = firrtl.subfield %c_x[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    %3 = firrtl.subfield %x[b] : !firrtl.bundle<a: uint<1>, b flip: uint<2>>
    firrtl.strictconnect %3, %2 : !firrtl.uint<2>
    firrtl.strictconnect %y, %c_y : !firrtl.uint<1>
  }
}

// -----

// Non-ground source: HW.

// CHECK-LABEL: "AggSourceHW"
firrtl.circuit "AggSourceHW" {
  // CHECK:      @Select
  // CHECK-NOT: out %out
  // CHECK-NEXT:   firrtl.subindex
  // CHECK-NEXT: }
  firrtl.module private @Select(in %in: !firrtl.vector<uint<1>, 5>,
                               out %out : !firrtl.uint<1>) {
    %sel = firrtl.subindex %in[3] : !firrtl.vector<uint<1>, 5>
    firrtl.strictconnect %out, %sel : !firrtl.uint<1>
  }
  // CHECK: @AggSourceHW
  firrtl.module @AggSourceHW(in %in : !firrtl.vector<uint<1>, 5>, out %out : !firrtl.uint<1>) {
    %s_in, %s_out = firrtl.instance s @Select(in in : !firrtl.vector<uint<1>, 5>,
                                              out out : !firrtl.uint<1>)
    // CHECK: firrtl.subindex
    firrtl.strictconnect %s_in, %in : !firrtl.vector<uint<1>, 5>
    firrtl.strictconnect %out, %s_out : !firrtl.uint<1>
  }
}

// -----

// Reject non-ground dest (for now).

// CHECK-LABEL: "AggHW"
firrtl.circuit "AggHW" {
  // CHECK:      module private @UTurn(in %in: !firrtl.vector<uint<1>, 5>, out %out
  // CHECK-NEXT:   firrtl.strictconnect
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.vector<uint<1>, 5>,
                               out %out : !firrtl.vector<uint<1>, 5>) {
    firrtl.strictconnect %out, %in : !firrtl.vector<uint<1>, 5>
  }
  firrtl.module @AggHW(in %in : !firrtl.vector<uint<1>, 5>, out %out : !firrtl.vector<uint<1>, 5>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.vector<uint<1>, 5>,
                                             out out : !firrtl.vector<uint<1>, 5>)
    firrtl.strictconnect %u_in, %in : !firrtl.vector<uint<1>, 5>
    firrtl.strictconnect %out, %u_out : !firrtl.vector<uint<1>, 5>
  }
}

// -----

// Reject multiple drivers: HW

// CHECK-LABEL: "MultiConnect"
firrtl.circuit "MultiConnect" {
  // CHECK:      module private @UTurn(in %in: !firrtl.uint<1>, out %out
  // CHECK-COUNT-2: strictconnect
  // CHECK-NEXT: }
  firrtl.module private @UTurn(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  firrtl.module @MultiConnect(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Annotations: DontTouch blocks

// CHECK-LABEL: "DontTouch"
firrtl.circuit "DontTouch" {
  // CHECK:      DontTouchAnnotation
  // CHECK-SAME: out %out
  // CHECK-NEXT:   firrtl.strictconnect
  firrtl.module private @UTurn(in %in: !firrtl.uint<1> [{class = "firrtl.transforms.DontTouchAnnotation"}],
                               out %out : !firrtl.uint<1>) {
    firrtl.strictconnect %out, %in : !firrtl.uint<1>
  }
  firrtl.module @DontTouch(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Annotations: hoist through, but leave on original.

// CHECK-LABEL: HWAnno
firrtl.circuit "HWAnno" {
  // CHECK:      module private @UAnno(in %in: !firrtl.uint<1>) {
  // CHECK-NEXT:   %[[W:.+]] = firrtl.wire {annotations
  // CHECK-NEXT:   firrtl.strictconnect %[[W]], %in
  // CHECK-NEXT: }
  firrtl.module private @UAnno(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %w = firrtl.wire {annotations = [{class = "circt.test"}]} : !firrtl.uint<1>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module @HWAnno(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    // CHECK: %[[HW_IN:.+]] = firrtl.instance
    %u_in, %u_out = firrtl.instance u @UAnno(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    // CHECK: strictconnect %out, %[[HW_IN]]
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// "Forceable" declarations.

// CHECK-LABEL: HWForceable
firrtl.circuit "HWForceable" {
  // CHECK:      module private @UForceable(in %in: !firrtl.uint<1>, out %out
  firrtl.module private @UForceable(in %in: !firrtl.uint<1>,
                               out %out : !firrtl.uint<1>) {
    %w, %w_ref = firrtl.wire forceable : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    firrtl.strictconnect %w, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %w : !firrtl.uint<1>
  }
  firrtl.module @HWForceable(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<1>) {
    %u_in, %u_out = firrtl.instance u @UForceable(in in : !firrtl.uint<1>,
                                             out out : !firrtl.uint<1>)
    firrtl.strictconnect %u_in, %in : !firrtl.uint<1>
    firrtl.strictconnect %out, %u_out : !firrtl.uint<1>
  }
}

// -----

// Multiple layers of HW aggregate.

// CHECK-LABEL: "AggAgg"
firrtl.circuit "AggAgg" {
  // CHECK-NOT: out %out
  firrtl.module private @IndexIndex(in %in : !firrtl.vector<vector<uint<1>, 5>, 5>, out %out : !firrtl.uint<1>) {
    %vec_sel = firrtl.subindex %in[1] : !firrtl.vector<vector<uint<1>, 5>, 5>
    %sel = firrtl.subindex %vec_sel[3] : !firrtl.vector<uint<1>, 5>
    firrtl.strictconnect %out, %sel : !firrtl.uint<1>
  }
  // CHECK: @AggAgg
  firrtl.module @AggAgg(in %in : !firrtl.vector<vector<uint<1>, 5>, 5>, out %out : !firrtl.uint<1>) {
    // CHECK: %[[IN:.+]] = firrtl.instance
    %ii_in, %ii_out = firrtl.instance ii @IndexIndex(in in : !firrtl.vector<vector<uint<1>, 5>, 5>, out out : !firrtl.uint<1>)
    firrtl.strictconnect %ii_in, %in : !firrtl.vector<vector<uint<1>, 5>, 5>
    // CHECK: %[[IN_1:.+]] = firrtl.subindex %[[IN]][1]
    // CHECK: %[[IN_1_3:.+]] = firrtl.subindex %[[IN_1]][3]
    // CHECK: strictconnect %out, %[[IN_1_3]]
    firrtl.strictconnect %out, %ii_out : !firrtl.uint<1>
  }
}

// -----
// Leave property signals as-is, for now.

// CHECK-LABEL: "SimpleProp"
firrtl.circuit "SimpleProp" {
  // CHECK:      module private @UTurn(in %in: !firrtl.string, out %out
  firrtl.module private @UTurn(in %in: !firrtl.string,
                               out %out : !firrtl.string) {
    firrtl.propassign %out, %in : !firrtl.string
  }
  firrtl.module @SimpleProp(in %in : !firrtl.string, out %out : !firrtl.string) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : !firrtl.string,
                                             out out : !firrtl.string)
    firrtl.propassign %u_in, %in : !firrtl.string
    firrtl.propassign %out, %u_out : !firrtl.string
  }
}

// -----
// Leave foreign types as-is, for now.

// CHECK-LABEL: "SimpleForeign"
firrtl.circuit "SimpleForeign" {
  // CHECK:      module private @UTurn(in %in: i1, out %out: i1)
  firrtl.module private @UTurn(in %in: i1,
                               out %out : i1) {
    firrtl.strictconnect %out, %in : i1
  }
  firrtl.module @SimpleForeign(in %in : i1, out %out : i1) {
    %u_in, %u_out = firrtl.instance u @UTurn(in in : i1,
                                             out out : i1)
    firrtl.strictconnect %u_in, %in : i1
    firrtl.strictconnect %out, %u_out : i1
  }
}
