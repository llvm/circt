// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-full-reset))' --split-input-file %s | FileCheck %s

// CHECK-LABEL: firrtl.circuit "Mem"
firrtl.circuit "Mem" {
  firrtl.module public @Mem(
      out %d: !firrtl.probe<vector<uint<8>, 8>>,
      out %d2: !firrtl.probe<vector<uint<8>, 8>>,
      in %reset: !firrtl.asyncreset
  ) attributes {
    portAnnotations = [[], [], [{class = "circt.FullResetAnnotation", resetType = "async"}]]
  } {
    %dbg, %mem_read, %mem_write, %debug = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["dbg", "read", "write", "debug"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.probe<vector<uint<8>, 8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>,
        !firrtl.probe<vector<uint<8>, 8>>
    firrtl.ref.define %d, %debug : !firrtl.probe<vector<uint<8>, 8>>
    firrtl.ref.define %d2, %dbg : !firrtl.probe<vector<uint<8>, 8>>
  }
  // CHECK: %mem_read = firrtl.wire
  // CHECK: %mem = firrtl.regreset
  // CHECK: firrtl.when
  // CHECK: %mem_write = firrtl.wire
  // CHECK: firrtl.ref.send %mem
}

// -----
// CHECK-LABEL: firrtl.circuit "GCTModule"
firrtl.circuit "GCTModule" {
  firrtl.module public @GCTModule(in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "async"}]]
  } {
    %rf_read, %rf_write = firrtl.mem Undefined {
      annotations = [
        {circt.fieldID = 1 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 1 : i64, type = "source"},
        {circt.fieldID = 1 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 2 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 2 : i64, type = "source"},
        {circt.fieldID = 2 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 3 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 3 : i64, type = "source"},
        {circt.fieldID = 3 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 4 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 4 : i64, type = "source"},
        {circt.fieldID = 4 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 5 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 5 : i64, type = "source"},
        {circt.fieldID = 5 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 6 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 6 : i64, type = "source"},
        {circt.fieldID = 6 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 7 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 7 : i64, type = "source"},
        {circt.fieldID = 7 : i64, class = "firrtl.transforms.DontTouchAnnotation"},
        {circt.fieldID = 8 : i64, class = "sifive.enterprise.grandcentral.ReferenceDataTapKey", id = 0 : i64, portID = 8 : i64, type = "source"},
        {circt.fieldID = 8 : i64, class = "firrtl.transforms.DontTouchAnnotation"}
      ],
      depth = 8 : i64,
      name = "rf",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
  }
  // CHECK: %rf = firrtl.regreset
  // CHECK-SAME: ReferenceDataTapKey
  // CHECK-SAME: DontTouchAnnotation
}

// -----
// CHECK-LABEL: firrtl.circuit "WriteMask"
firrtl.circuit "WriteMask" {
  firrtl.module public @WriteMask(in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "async"}]]
  } {
    %mem_read, %mem_write = firrtl.mem Undefined {
      depth = 8 : i64,
      name = "mem",
      portNames = ["read", "write"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: vector<uint<8>, 2>>,
        !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: vector<uint<8>, 2>, mask: vector<uint<1>, 2>>
  }
  // CHECK: %mem = firrtl.regreset
  // CHECK-SAME: !firrtl.vector<vector<uint<8>, 2>, 8>
  // CHECK: %mem_write = firrtl.wire
  // CHECK: firrtl.when
}

// -----
// CHECK-LABEL: firrtl.circuit "NLA"
firrtl.circuit "NLA" {
  // CHECK: hw.hierpath private @path_old [@NLA::@foo, @Foo::@old]
  // CHECK: hw.hierpath private @path_new [@NLA::@foo, @Foo]
  hw.hierpath private @path_old [@NLA::@foo, @Foo::@old]
  hw.hierpath private @path_new [@NLA::@foo, @Foo]
  // CHECK-LABEL: firrtl.module private @Foo
  firrtl.module private @Foo() {
    // CHECK: %old = firrtl.regreset sym @old
    // CHECK-SAME: {circt.nonlocal = @path, class = "oldNLA"}
    %old_r = firrtl.mem sym @old Undefined {
      annotations = [{circt.nonlocal = @path, class = "oldNLA"}],
      depth = 4 : i64,
      name = "old",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>
    // CHECK: %new = firrtl.regreset
    // CHECK-SAME: {circt.nonlocal = @path, class = "newNLA"}
    %new_r = firrtl.mem Undefined {
      annotations = [{circt.nonlocal = @path, class = "newNLA"}],
      depth = 4 : i64,
      name = "new",
      portNames = ["r"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<32>>
  }
  firrtl.module public @NLA(in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "async"}]]
  } {
    firrtl.instance foo sym @foo @Foo()
  }
}

// -----
// Sequential memories stay memories even under async FullReset.
// CHECK-LABEL: firrtl.circuit "SkipMemoryMacros"
firrtl.circuit "SkipMemoryMacros" {
  firrtl.module @SkipMemoryMacros(in %reset: !firrtl.asyncreset) attributes {
    portAnnotations = [[{class = "circt.FullResetAnnotation", resetType = "async"}]]
  } {
    // CHECK-COUNT-4: firrtl.mem
    %latency_1r1w = firrtl.mem Undefined {
      depth = 2 : i64, name = "m", portNames = ["rw"],
      readLatency = 1 : i32, writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
    %latency_1r2w = firrtl.mem Undefined {
      depth = 2 : i64, name = "m", portNames = ["rw"],
      readLatency = 1 : i32, writeLatency = 2 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
    %latency_2r1w = firrtl.mem Undefined {
      depth = 2 : i64, name = "m", portNames = ["rw"],
      readLatency = 2 : i32, writeLatency = 1 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
    %latency_4r4w = firrtl.mem Undefined {
      depth = 2 : i64, name = "m", portNames = ["rw"],
      readLatency = 4 : i32, writeLatency = 4 : i32
    } : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, rdata flip: uint<1>, wmode: uint<1>, wdata: uint<1>, wmask: uint<1>>
  }
}
