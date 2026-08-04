// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// Basic async full-reset: reset-less register becomes regreset.
// CHECK-LABEL: firrtl.module @AsyncFullReset
firrtl.circuit "AsyncFullReset" {
  firrtl.module @AsyncFullReset(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}],
      in %in: !firrtl.uint<8>) {
    // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
// Exclude annotation is consumed; registers stay reset-less.
// CHECK-LABEL: firrtl.module @Excluded
// CHECK-NOT: ExcludeFromFullResetAnnotation
// CHECK: %reg = firrtl.reg %clock
firrtl.circuit "Excluded" {
  firrtl.module @Excluded(in %clock: !firrtl.clock, in %in: !firrtl.uint<8>)
      attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg, %in : !firrtl.uint<8>
  }
}

// -----
// Child inherits async domain; reset is wired through an added port.
// CHECK-LABEL: firrtl.module @Child
// CHECK-SAME: in %reset: !firrtl.asyncreset
// CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
// CHECK-LABEL: firrtl.module @Nested
// CHECK: firrtl.matchingconnect %child_reset, %reset
firrtl.circuit "Nested" {
  firrtl.module @Child(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  firrtl.module @Nested(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %child_clock = firrtl.instance child @Child(in clock: !firrtl.clock)
    firrtl.matchingconnect %child_clock, %clock : !firrtl.clock
  }
}

// -----
// Comb mems in async full-reset domains become resettable registers.
// CHECK-LABEL: firrtl.module @AsyncDomainMem
// CHECK-NOT: firrtl.mem
// CHECK: firrtl.regreset
firrtl.circuit "AsyncDomainMem" {
  firrtl.module @AsyncDomainMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.asyncreset
          [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----
// Sync full-reset domains keep comb mems.
// CHECK-LABEL: firrtl.module @SyncDomainMem
// CHECK: firrtl.mem
// CHECK-NOT: firrtl.reg
firrtl.circuit "SyncDomainMem" {
  firrtl.module @SyncDomainMem(
      in %clock: !firrtl.clock,
      in %reset: !firrtl.uint<1>
          [{class = "circt.FullResetAnnotation", resetType = "sync"}]) {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 4 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
}

// -----
// AsyncReset-less registers should inherit the annotated async reset signal.
firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %reg1 = firrtl.regreset sym @reg1 %clock, %extraReset, %c0_ui8
    %reg1 = firrtl.reg sym @reg1 %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg1, %in : !firrtl.uint<8>

    // Existing async reset remains untouched.
    // CHECK: %reg2 = firrtl.regreset %clock, %reset, %c1_ui8
    %reg2 = firrtl.regreset %clock, %reset, %c1_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %reg2, %in : !firrtl.uint<8>

    // Existing sync reset is moved to mux.
    // CHECK: %reg3 = firrtl.regreset %clock, %extraReset, %c0_ui8
    // CHECK: %0 = firrtl.mux(%init, %c1_ui8, %reg3)
    // CHECK: %1 = firrtl.mux(%init, %c1_ui8, %in)
    // CHECK: firrtl.matchingconnect %reg3, %1
    %reg3 = firrtl.regreset %clock, %init, %c1_ui8 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %reg3, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subfield op.
    // CHECK: %reg4 = firrtl.regreset %clock, %extraReset, %2
    // CHECK: %4 = firrtl.mux(%init, %reset4, %reg4)
    // CHECK: %5 = firrtl.subfield %reset4[a]
    // CHECK: %6 = firrtl.subfield %reg4[a]
    // CHECK: %7 = firrtl.mux(%init, %5, %in)
    // CHECK: firrtl.matchingconnect %6, %7
    %reset4 = firrtl.wire : !firrtl.bundle<a: uint<8>>
    %reg4 = firrtl.regreset %clock, %init, %reset4 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>
    %0 = firrtl.subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
    firrtl.matchingconnect %0, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subindex op.
    // CHECK: %reg5 = firrtl.regreset %clock, %extraReset, %8
    // CHECK: %10 = firrtl.mux(%init, %reset5, %reg5)
    // CHECK: firrtl.matchingconnect %reg5, %10
    // CHECK: %11 = firrtl.subindex %reset5[0]
    // CHECK: %12 = firrtl.subindex %reg5[0]
    // CHECK: %13 = firrtl.mux(%init, %11, %in)
    // CHECK: firrtl.matchingconnect %12, %13
    %reset5 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg5 = firrtl.regreset %clock, %init, %reset5 : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %1 = firrtl.subindex %reg5[0] : !firrtl.vector<uint<8>, 1>
    firrtl.matchingconnect %1, %in : !firrtl.uint<8>

    // Factoring of sync reset into mux works through subaccess op.
    // CHECK: %reg6 = firrtl.regreset %clock, %extraReset, %14
    // CHECK: %16 = firrtl.mux(%init, %reset6, %reg6)
    // CHECK: firrtl.matchingconnect %reg6, %16
    // CHECK: %17 = firrtl.subaccess %reset6[%in]
    // CHECK: %18 = firrtl.subaccess %reg6[%in]
    // CHECK: %19 = firrtl.mux(%init, %17, %in)
    // CHECK: firrtl.matchingconnect %18, %19
    %reset6 = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg6 = firrtl.regreset %clock, %init, %reset6 : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<8>, 1>, !firrtl.vector<uint<8>, 1>
    %2 = firrtl.subaccess %reg6[%in] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    firrtl.matchingconnect %2, %in : !firrtl.uint<8>

    // Subfields that are never assigned to should not leave unused reset
    // subfields behind.
    // CHECK-NOT: firrtl.subfield %reset4[a]
    // CHECK: %20 = firrtl.subfield %reg4[a]
    %3 = firrtl.subfield %reg4[a] : !firrtl.bundle<a: uint<8>>
  }
}
// -----
// Reset-less registers should inherit the annotated sync reset signal.
firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.uint<1> ) attributes {
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "circt.FullResetAnnotation", resetType = "sync"}]]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
    // CHECK: %reg1 = firrtl.regreset sym @reg1 %clock, %extraReset, %c0_ui8
    %reg1 = firrtl.reg sym @reg1 %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %reg1, %in : !firrtl.uint<8>

    // Existing async reset remains untouched.
    // CHECK: %reg2 = firrtl.regreset %clock, %reset, %c1_ui8
    %reg2 = firrtl.regreset %clock, %reset, %c1_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %reg2, %in : !firrtl.uint<8>

    // Existing sync reset remains untouched.
    // CHECK: %reg3 = firrtl.regreset %clock, %init, %c1_ui8
    %reg3 = firrtl.regreset %clock, %init, %c1_ui8 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %reg3, %in : !firrtl.uint<8>
  }
}
// -----
// Async reset inference should be able to construct reset values for aggregate
// types.
firrtl.circuit "Top" {
  // CHECK-LABEL: firrtl.module @Top
  firrtl.module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    // CHECK: %c0_ui = firrtl.constant 0 : !firrtl.const.uint
    // CHECK: %reg_uint = firrtl.regreset %clock, %reset, %c0_ui
    %reg_uint = firrtl.reg %clock : !firrtl.clock, !firrtl.uint
    // CHECK: %c0_si = firrtl.constant 0 : !firrtl.const.sint
    // CHECK: %reg_sint = firrtl.regreset %clock, %reset, %c0_si
    %reg_sint = firrtl.reg %clock : !firrtl.clock, !firrtl.sint
    // CHECK: %0 = firrtl.wire : !firrtl.const.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %c0_ui8 = firrtl.constant 0 : !firrtl.const.uint<8>
    // CHECK: %1 = firrtl.subfield %0[a]
    // CHECK: firrtl.matchingconnect %1, %c0_ui8
    // CHECK: %2 = firrtl.wire : !firrtl.const.bundle<x: uint<8>, y: uint<8>>
    // CHECK: %3 = firrtl.subfield %2[x]
    // CHECK: firrtl.matchingconnect %3, %c0_ui8
    // CHECK: %4 = firrtl.subfield %2[y]
    // CHECK: firrtl.matchingconnect %4, %c0_ui8
    // CHECK: %5 = firrtl.subfield %0[b]
    // CHECK: firrtl.matchingconnect %5, %2
    // CHECK: %reg_bundle = firrtl.regreset %clock, %reset, %0
    %reg_bundle = firrtl.reg %clock : !firrtl.clock, !firrtl.bundle<a: uint<8>, b: bundle<x: uint<8>, y: uint<8>>>
    // CHECK: %6 = firrtl.wire : !firrtl.const.vector<uint<8>, 4>
    // CHECK: %c0_ui8_0 = firrtl.constant 0 : !firrtl.const.uint<8>
    // CHECK: %7 = firrtl.subindex %6[0]
    // CHECK: firrtl.matchingconnect %7, %c0_ui8_0
    // CHECK: %8 = firrtl.subindex %6[1]
    // CHECK: firrtl.matchingconnect %8, %c0_ui8_0
    // CHECK: %9 = firrtl.subindex %6[2]
    // CHECK: firrtl.matchingconnect %9, %c0_ui8_0
    // CHECK: %10 = firrtl.subindex %6[3]
    // CHECK: firrtl.matchingconnect %10, %c0_ui8_0
    // CHECK: %reg_vector = firrtl.regreset %clock, %reset, %6
    %reg_vector = firrtl.reg %clock : !firrtl.clock, !firrtl.vector<uint<8>, 4>
    // CHECK: [[ENUMCREATE:%[0-9]+]] = firrtl.enumcreate a(%c0_ui0) : (!firrtl.const.uint<0>) -> !firrtl.const.enum<a>
    // CHECK: %reg_enum_0 = firrtl.regreset %clock, %reset, [[ENUMCREATE]] : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.enum<a>, !firrtl.enum<a>
    %reg_enum_0 = firrtl.reg %clock : !firrtl.clock, !firrtl.enum<a>
    // CHECK: [[BITCAST:%[0-9]+]] = firrtl.bitcast %c0_ui1 : (!firrtl.const.uint<1>) -> !firrtl.const.enum<a = 1>
    // CHECK: %reg_enum_1 = firrtl.regreset %clock, %reset, [[BITCAST]] : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.enum<a = 1>, !firrtl.enum<a = 1>
    %reg_enum_1 = firrtl.reg %clock : !firrtl.clock, !firrtl.enum<a = 1>
  }
}
// -----
// Reset should reuse ports if name and type matches for async wiring.
firrtl.circuit "ReusePortsAsync" {
  // CHECK-LABEL: firrtl.module @Child
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @Child(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @BadName
  // CHECK-SAME: in %reset: !firrtl.asyncreset,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %existingReset: !firrtl.asyncreset
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @BadName(in %clock: !firrtl.clock, in %existingReset: !firrtl.asyncreset) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ReusePorts
  firrtl.module @ReusePortsAsync(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    // CHECK: %child_clock, %child_reset = firrtl.instance child
    // CHECK: firrtl.matchingconnect %child_reset, %reset
    // CHECK: %badName_reset, %badName_clock, %badName_existingReset = firrtl.instance badName
    // CHECK: firrtl.matchingconnect %badName_reset, %reset
    %child_clock, %child_reset = firrtl.instance child @Child(in clock: !firrtl.clock, in reset: !firrtl.asyncreset)
    %badName_clock, %badName_existingReset = firrtl.instance badName @BadName(in clock: !firrtl.clock, in existingReset: !firrtl.asyncreset)
  }
}
// -----
// Reset should reuse ports if name and type matches for sync wiring.
firrtl.circuit "ReusePortsSync" {
  // CHECK-LABEL: firrtl.module @Child
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %reset: !firrtl.uint<1>
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @Child(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @BadName
  // CHECK-SAME: in %reset: !firrtl.uint<1>,
  // CHECK-SAME: in %clock: !firrtl.clock
  // CHECK-SAME: in %existingReset: !firrtl.uint<1>
  // CHECK: %reg = firrtl.regreset %clock, %reset, %c0_ui8
  firrtl.module @BadName(in %clock: !firrtl.clock, in %existingReset: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @ReusePorts
  firrtl.module @ReusePortsSync(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>) attributes {
    portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "sync"}]]} {
    // CHECK: %child_clock, %child_reset = firrtl.instance child
    // CHECK: firrtl.matchingconnect %child_reset, %reset
    // CHECK: %badName_reset, %badName_clock, %badName_existingReset = firrtl.instance badName
    // CHECK: firrtl.matchingconnect %badName_reset, %reset
    %child_clock, %child_reset = firrtl.instance child @Child(in clock: !firrtl.clock, in reset: !firrtl.uint<1>)
    %badName_clock, %badName_existingReset = firrtl.instance badName @BadName(in clock: !firrtl.clock, in existingReset: !firrtl.uint<1>)
  }
}
// -----
// Infer async reset: nested
firrtl.circuit "FullAsyncNested" {
  // CHECK-LABEL: firrtl.module @FullAsyncNestedDeeper
  firrtl.module @FullAsyncNestedDeeper(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    // CHECK: %io_out_REG = firrtl.regreset %clock, %reset, %c1_ui1
    %io_out_REG = firrtl.regreset %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<8>
    firrtl.matchingconnect %io_out_REG, %io_in : !firrtl.uint<8>
    firrtl.matchingconnect %io_out, %io_out_REG : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncNestedChild
  firrtl.module @FullAsyncNestedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncNestedDeeper(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %inst_reset, %reset : !firrtl.asyncreset
    firrtl.matchingconnect %inst_io_in, %io_in : !firrtl.uint<8>
    // CHECK: %io_out_REG = firrtl.regreset %clock, %reset, %c0_ui8
    %io_out_REG = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %io_out_REG, %io_in : !firrtl.uint<8>
    %0 = firrtl.add %io_out_REG, %inst_io_out : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
    %1 = firrtl.bits %0 7 to 0 : (!firrtl.uint<9>) -> !firrtl.uint<8>
    firrtl.matchingconnect %io_out, %1 : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncNested
  firrtl.module @FullAsyncNested(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {
    portAnnotations=[[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "circt.FullResetAnnotation", resetType = "async"}], [], []] } {
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncNestedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %inst_reset, %reset : !firrtl.asyncreset
    firrtl.matchingconnect %io_out, %inst_io_out : !firrtl.uint<8>
    firrtl.matchingconnect %inst_io_in, %io_in : !firrtl.uint<8>
  }
}
// -----
// Infer async reset: excluded
// TODO: Check that no extraReset port present
firrtl.circuit "FullAsyncExcluded" {
  // CHECK-LABEL: firrtl.module @FullAsyncExcludedChild
  // CHECK-SAME: (in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>)
  firrtl.module @FullAsyncExcludedChild(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>) attributes {annotations = [{class = "circt.ExcludeFromFullResetAnnotation"}]} {
    // CHECK: %io_out_REG = firrtl.reg %clock
    %io_out_REG = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
    firrtl.matchingconnect %io_out_REG, %io_in : !firrtl.uint<8>
    firrtl.matchingconnect %io_out, %io_out_REG : !firrtl.uint<8>
  }
  // CHECK-LABEL: firrtl.module @FullAsyncExcluded
  firrtl.module @FullAsyncExcluded(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_in: !firrtl.uint<8>, out %io_out: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset) attributes {
     portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    // CHECK: %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncExcludedChild
    %inst_clock, %inst_reset, %inst_io_in, %inst_io_out = firrtl.instance inst @FullAsyncExcludedChild(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in io_in: !firrtl.uint<8>, out io_out: !firrtl.uint<8>)
    firrtl.matchingconnect %inst_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %inst_reset, %reset : !firrtl.asyncreset
    firrtl.matchingconnect %io_out, %inst_io_out : !firrtl.uint<8>
    firrtl.matchingconnect %inst_io_in, %io_in : !firrtl.uint<8>
  }
}
// -----

// Local wire as async reset should be moved before all its uses.
firrtl.circuit "WireShouldDominate" {
  // CHECK-LABEL: firrtl.module @WireShouldDominate
  firrtl.module @WireShouldDominate(in %clock: !firrtl.clock) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %localReset = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
  }
}
// -----

// Local node as async reset should be moved before all its uses if its input
// value dominates the target location in the module.
firrtl.circuit "MovableNodeShouldDominate" {
  // CHECK-LABEL: firrtl.module @MovableNodeShouldDominate
  firrtl.module @MovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // does not block move of node
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %0 = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: %localReset = firrtl.node sym @theReset %0
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
  }
}
// -----

// Local node as async reset should be replaced by a wire and moved before all
// its uses if its input value does not dominate the target location in the
// module.
firrtl.circuit "UnmovableNodeShouldDominate" {
  // CHECK-LABEL: firrtl.module @UnmovableNodeShouldDominate
  firrtl.module @UnmovableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire sym @theReset
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: %0 = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: %1 = firrtl.node %0 :
    // CHECK-NEXT: firrtl.matchingconnect %localReset, %1 :
  }
}
// -----

// Same test as above, ensure works w/forceable node.
firrtl.circuit "UnmovableForceableNodeShouldDominate" {
  // CHECK-LABEL: firrtl.module @UnmovableForceableNodeShouldDominate
  firrtl.module @UnmovableForceableNodeShouldDominate(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset, %ref = firrtl.node sym @theReset %0 forceable {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset, %{{.+}} = firrtl.wire sym @theReset
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: %0 = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: %1:2 = firrtl.node %0 forceable
    // CHECK-NEXT: firrtl.matchingconnect %localReset, %1#0
  }
}
// -----

// Move of local async resets should work across blocks.
firrtl.circuit "MoveAcrossBlocks1" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks1
  firrtl.module @MoveAcrossBlocks1(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 : !firrtl.uint<1> {
      %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    firrtl.when %ui1 : !firrtl.uint<1> {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   [[TMP2:%.+]] = firrtl.node [[TMP]] : !firrtl.asyncreset
    // CHECK-NEXT:   firrtl.matchingconnect %localReset, [[TMP2]]
    // CHECK-NEXT: }
  }
}
// -----

firrtl.circuit "MoveAcrossBlocks2" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks2
  firrtl.module @MoveAcrossBlocks2(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 : !firrtl.uint<1> {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    }
    firrtl.when %ui1 : !firrtl.uint<1> {
      %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   [[TMP2:%.+]] = firrtl.node [[TMP]] : !firrtl.asyncreset
    // CHECK-NEXT:   firrtl.matchingconnect %localReset, [[TMP2]]
    // CHECK-NEXT: }
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
  }
}
// -----

firrtl.circuit "MoveAcrossBlocks3" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks3
  firrtl.module @MoveAcrossBlocks3(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    firrtl.when %ui1 : !firrtl.uint<1> {
      %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
      %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    }
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT:   [[TMP2:%.+]] = firrtl.node [[TMP]] : !firrtl.asyncreset
    // CHECK-NEXT:   firrtl.matchingconnect %localReset, [[TMP2]]
    // CHECK-NEXT: }
  }
}
// -----

firrtl.circuit "MoveAcrossBlocks4" {
  // CHECK-LABEL: firrtl.module @MoveAcrossBlocks4
  firrtl.module @MoveAcrossBlocks4(in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
    // <-- should move reset here
    firrtl.when %ui1 : !firrtl.uint<1> {
      %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8> // gets wired to localReset
    }
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset // blocks move of node
    %localReset = firrtl.node sym @theReset %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK-NEXT: %localReset = firrtl.wire
    // CHECK-NEXT: firrtl.when %ui1 : !firrtl.uint<1> {
    // CHECK-NEXT:   [[RV:%.+]] = firrtl.constant 0
    // CHECK-NEXT:   %reg = firrtl.regreset %clock, %localReset, [[RV]]
    // CHECK-NEXT: }
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.asAsyncReset %ui1
    // CHECK-NEXT: [[TMP2:%.+]] = firrtl.node [[TMP]] : !firrtl.asyncreset
    // CHECK-NEXT: firrtl.matchingconnect %localReset, [[TMP2]]
  }
}
// -----

firrtl.circuit "SubAccess" {
  firrtl.module @SubAccess(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %init: !firrtl.uint<1>, in %in: !firrtl.uint<8>, in %extraReset: !firrtl.asyncreset ) attributes {
    // CHECK-LABEL: firrtl.module @SubAccess
    portAnnotations = [[],[],[],[],[{class = "firrtl.transforms.DontTouchAnnotation"}, {class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<2>
    %arr = firrtl.wire : !firrtl.vector<uint<8>, 1>
    %reg6 = firrtl.regreset %clock, %init, %c1_ui8 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>
    %2 = firrtl.subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    firrtl.matchingconnect %2, %in : !firrtl.uint<8>
    // CHECK:  %reg6 = firrtl.regreset %clock, %extraReset, %c0_ui2  : !firrtl.clock, !firrtl.asyncreset, !firrtl.const.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: %0 = firrtl.mux(%init, %c1_ui2, %reg6)
    // CHECK: firrtl.matchingconnect %reg6, %0
    // CHECK-NEXT:  %[[v0:.+]] = firrtl.subaccess %arr[%reg6] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<2>
    // CHECK-NEXT:  firrtl.matchingconnect %[[v0]], %in : !firrtl.uint<8>

  }
}
// -----

// This is a regression check to ensure that a zero-width register gets a proper
// reset value.
// CHECK-LABEL: firrtl.module @ZeroWidthRegister
firrtl.circuit "ZeroWidthRegister" {
  firrtl.module @ZeroWidthRegister(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[],[{class = "circt.FullResetAnnotation", resetType = "async"}]]} {
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<0>
    // CHECK-NEXT: [[TMP:%.+]] = firrtl.constant 0 : !firrtl.const.uint<0>
    // CHECK-NEXT: %reg = firrtl.regreset %clock, %reset, [[TMP]]
  }
}
// -----

// Every module which is contained inside a reset domain should be annotated as
// such, so that internal registers can be lowered later correctly.
// https://github.com/llvm/circt/issues/7675
firrtl.circuit "top" {
  // CHECK: firrtl.module private @test
  // CHECK-SAME: annotations = [{class = "circt.FullResetAnnotation"}]
  firrtl.module private @test(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
    %resetvalue = firrtl.wire : !firrtl.uint<8>
    %invalid_ui8 = firrtl.invalidvalue : !firrtl.uint<8>
    firrtl.matchingconnect %resetvalue, %invalid_ui8 : !firrtl.uint<8>
    %reg1 = firrtl.regreset %clock, %reset, %resetvalue : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
    firrtl.matchingconnect %reg1, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %reg1 : !firrtl.uint<8>
  }
  // CHECK: firrtl.module @top
  // CHECK-SAME: annotations = [{class = "circt.FullResetAnnotation"}]
  firrtl.module @top(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}], in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) attributes {convention = #firrtl<convention scalarized>} {
    %child_clock, %child_reset, %child_in, %child_out = firrtl.instance child @test(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, in in: !firrtl.uint<8>, out out: !firrtl.uint<8>)
    firrtl.matchingconnect %child_clock, %clock : !firrtl.clock
    firrtl.matchingconnect %child_reset, %reset : !firrtl.asyncreset
    firrtl.matchingconnect %child_in, %in : !firrtl.uint<8>
    firrtl.matchingconnect %out, %child_out : !firrtl.uint<8>
  }
}
// -----

// CHECK-LABEL: "MovableNodeShouldDominateInstance"
firrtl.circuit "MovableNodeShouldDominateInstance" {
  firrtl.module @MovableNodeShouldDominateInstance(in %clock: !firrtl.clock) {
    %child_clock = firrtl.instance child @Child(in clock: !firrtl.clock)
    firrtl.connect %child_clock, %clock : !firrtl.clock
    %ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.asAsyncReset %ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %localReset = firrtl.node %0 {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK:       %localReset = firrtl.wire {annotations = [{class = "circt.FullResetAnnotation", resetType = "async"}]} : !firrtl.asyncreset
    // CHECK:       %child_localReset, %child_clock = firrtl.instance child @Child(in localReset: !firrtl.asyncreset, in clock: !firrtl.clock
  }
  firrtl.module @Child(in %clock: !firrtl.clock) {
    // CHECK: firrtl.regreset %clock, %localReset, %c0_ui8
    %reg = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
}
// -----
// A module should be able to be in two reset domains as long as both domains
// share the same port name and type.

firrtl.circuit "top" {
  firrtl.module @child(in %clock: !firrtl.clock) {
    %r = firrtl.reg %clock : !firrtl.clock, !firrtl.uint<8>
  }
  firrtl.module @parent_1(in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %c_clock  = firrtl.instance c @child(in clock: !firrtl.clock)
  }
  firrtl.module @parent_2(in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    %c_clock = firrtl.instance c @child(in clock: !firrtl.clock)
  }
  firrtl.module @top(in %clock: !firrtl.clock) {
    %parent_1_reset = firrtl.instance p1 @parent_1(in reset: !firrtl.asyncreset)
    %parent_2_reset = firrtl.instance p2 @parent_2(in reset: !firrtl.asyncreset)
  }
}
// -----
// Issue 9396
firrtl.circuit "Foo" {
  // CHECK-LABEL: firrtl.module @Baz
  firrtl.module @Baz(in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    firrtl.instance bar @Bar()
    // CHECK: firrtl.matchingconnect %bar_reset, %reset : !firrtl.asyncreset
  }
  // CHECK-LABEL: firrtl.module private @Bar(in %reset: !firrtl.asyncreset)
  firrtl.module private @Bar() {
  }
  // CHECK-LABEL: firrtl.module @Foo(in %reset: !firrtl.asyncreset
  firrtl.module @Foo(in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    firrtl.instance bar @Bar()
    // CHECK: firrtl.matchingconnect %bar_reset, %reset : !firrtl.asyncreset
  }
}
// -----
// Allow modules with full reset to be instantiated outside of a reset domain,
// with reset tied-off. (Async reset variant)
// See https://github.com/llvm/circt/issues/9396
firrtl.circuit "AsyncNoDomainTieOff" {
  // CHECK-LABEL: firrtl.module @AsyncNoDomainTieOff
  firrtl.module @AsyncNoDomainTieOff(in %reset: !firrtl.asyncreset [{class = "circt.FullResetAnnotation", resetType = "async"}]) {
    // CHECK: %foo_reset = firrtl.instance foo @AsyncFoo(in reset: !firrtl.asyncreset)
    // CHECK-NEXT: firrtl.matchingconnect %foo_reset, %reset
    firrtl.instance foo @AsyncFoo()
  }
  // CHECK-LABEL: firrtl.module @AsyncTopB
  firrtl.module @AsyncTopB() {
    // CHECK: %foo_reset = firrtl.instance foo @AsyncFoo(in reset: !firrtl.asyncreset)
    // CHECK-NEXT: %c0_asyncreset = firrtl.specialconstant 0 : !firrtl.asyncreset
    // CHECK-NEXT: firrtl.matchingconnect %foo_reset, %c0_asyncreset
    firrtl.instance foo @AsyncFoo()
  }
  // CHECK-LABEL: firrtl.module private @AsyncFoo
  // CHECK-SAME: in %reset: !firrtl.asyncreset
  firrtl.module private @AsyncFoo() {
    %0 = firrtl.specialconstant false : !firrtl.clock
    %1 = firrtl.reg %0 : !firrtl.clock, !firrtl.uint<42>
  }
}
// -----
// Allow modules with full reset to be instantiated outside of a reset domain,
// with reset tied-off. (Sync reset variant)
// See https://github.com/llvm/circt/issues/9396
firrtl.circuit "SyncNoDomainTieOff" {
  // CHECK-LABEL: firrtl.module @SyncNoDomainTieOff
  firrtl.module @SyncNoDomainTieOff(in %reset: !firrtl.uint<1> [{class = "circt.FullResetAnnotation", resetType = "sync"}]) {
    // CHECK: %foo_reset = firrtl.instance foo @SyncFoo(in reset: !firrtl.uint<1>)
    // CHECK-NEXT: firrtl.matchingconnect %foo_reset, %reset
    firrtl.instance foo @SyncFoo()
  }
  // CHECK-LABEL: firrtl.module @SyncTopB
  firrtl.module @SyncTopB() {
    // CHECK: %foo_reset = firrtl.instance foo @SyncFoo(in reset: !firrtl.uint<1>)
    // CHECK-NEXT: %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.matchingconnect %foo_reset, %c0_ui1
    firrtl.instance foo @SyncFoo()
  }
  // CHECK-LABEL: firrtl.module private @SyncFoo
  // CHECK-SAME: in %reset: !firrtl.uint<1>
  firrtl.module private @SyncFoo() {
    %0 = firrtl.specialconstant false : !firrtl.clock
    %1 = firrtl.reg %0 : !firrtl.clock, !firrtl.uint<42>
  }
}
