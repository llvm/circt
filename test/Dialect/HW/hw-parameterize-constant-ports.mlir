// RUN: circt-opt -hw-parameterize-constant-ports %s | FileCheck %s

// Test multiple constant ports and mixed constant/variable ports
// mode1 is constant in inst0 but variable in inst1, so it should NOT be parameterized
// id and mode2 are constant in both instances, so they should be parameterized
// CHECK-LABEL: hw.module private @MultipleConstantPorts<id: i8, mode2: i4>
// CHECK-SAME: (in %mode1 : i4, in %data : i32, out out : i32)
// CHECK: %[[ID:[0-9]+]] = hw.param.value i8 = #hw.param.decl.ref<"id">
// CHECK: %[[MODE2:[0-9]+]] = hw.param.value i4 = #hw.param.decl.ref<"mode2">
// CHECK: dbg.variable "id", %[[ID]] : i8
// CHECK: dbg.variable "mode1", %mode1 : i4
// CHECK: dbg.variable "mode2", %[[MODE2]] : i4
// CHECK: hw.output %data : i32
hw.module private @MultipleConstantPorts(in %id: i8, in %mode1: i4, in %mode2: i4, in %data: i32, out out: i32) {
  dbg.variable "id", %id : i8
  dbg.variable "mode1", %mode1 : i4
  dbg.variable "mode2", %mode2 : i4
  hw.output %data : i32
}

// CHECK-LABEL: hw.module @UseMultipleConstantPorts
hw.module @UseMultipleConstantPorts(in %data: i32, in %mode: i4, out out0: i32, out out1: i32) {
  %c0_i8 = hw.constant 0 : i8
  %c0_i4 = hw.constant 0 : i4

  // CHECK: %inst0.out = hw.instance "inst0" @MultipleConstantPorts<id: i8 = 0, mode2: i4 = 0>
  // CHECK-SAME: (mode1: %c0_i4: i4, data: %data: i32) -> (out: i32)
  %inst0.out = hw.instance "inst0" @MultipleConstantPorts(id: %c0_i8: i8, mode1: %c0_i4: i4, mode2: %c0_i4: i4, data: %data: i32) -> (out: i32)

  %c1_i8 = hw.constant 1 : i8
  %c1_i4 = hw.constant 1 : i4
  // CHECK: %inst1.out = hw.instance "inst1" @MultipleConstantPorts<id: i8 = 1, mode2: i4 = 1>
  // CHECK-SAME: (mode1: %mode: i4, data: %data: i32) -> (out: i32)
  %inst1.out = hw.instance "inst1" @MultipleConstantPorts(id: %c1_i8: i8, mode1: %mode: i4, mode2: %c1_i4: i4, data: %data: i32) -> (out: i32)

  hw.output %inst0.out, %inst1.out : i32, i32
}

// Test that public modules are not parameterized
// CHECK-LABEL: hw.module @PublicModule(in %id : i32, in %clk : i1, out out : i32)
// CHECK-NOT: hw.param.value
hw.module @PublicModule(in %id: i32, in %clk: i1, out out: i32) {
  hw.output %id : i32
}

// CHECK-LABEL: hw.module @UsePublicModule
hw.module @UsePublicModule(in %clk: i1, out out: i32) {
  %c0_i32 = hw.constant 0 : i32
  
  // CHECK: %inst.out = hw.instance "inst" @PublicModule(id: %c0_i32: i32, clk: %clk: i1) -> (out: i32)
  %inst.out = hw.instance "inst" @PublicModule(id: %c0_i32: i32, clk: %clk: i1) -> (out: i32)
  
  hw.output %inst.out : i32
}

// The leaf module should be parameterized first, then the middle module
// CHECK-LABEL: hw.module private @Leaf<id: i8>
// CHECK-SAME: (out out : i8)
// CHECK: hw.param.value i8 = #hw.param.decl.ref<"id">
// CHECK: hw.output
hw.module private @Leaf(in %id: i8, out out: i8) {
  hw.output %id : i8
}

// CHECK-LABEL: hw.module private @Middle<mid_id: i8>
// CHECK-SAME: (out out : i8)
// CHECK: %leaf.out = hw.instance "leaf" @Leaf<id: i8 = #hw.param.decl.ref<"mid_id">>() -> (out: i8)
// CHECK: hw.output %leaf.out : i8
hw.module private @Middle(in %mid_id: i8, out out: i8) {
  %leaf.out = hw.instance "leaf" @Leaf(id: %mid_id: i8) -> (out: i8)
  hw.output %leaf.out : i8
}

// CHECK-LABEL: hw.module @UseHierarchy
hw.module @UseHierarchy(out out0: i8, out out1: i8) {
  %c10_i8 = hw.constant 10 : i8
  %c20_i8 = hw.constant 20 : i8

  // CHECK: %mid0.out = hw.instance "mid0" @Middle<mid_id: i8 = 10>() -> (out: i8)
  %mid0.out = hw.instance "mid0" @Middle(mid_id: %c10_i8: i8) -> (out: i8)

  // CHECK: %mid1.out = hw.instance "mid1" @Middle<mid_id: i8 = 20>() -> (out: i8)
  %mid1.out = hw.instance "mid1" @Middle(mid_id: %c20_i8: i8) -> (out: i8)

  hw.output %mid0.out, %mid1.out : i8, i8
}

// Test with existing parameters
// CHECK-LABEL: hw.module private @WithExistingParams<WIDTH: i32, id: i32>
// CHECK-SAME: (in %data : !hw.int<#hw.param.decl.ref<"WIDTH">>, out out : i32)
// CHECK: hw.param.value i32 = #hw.param.decl.ref<"id">
hw.module private @WithExistingParams<WIDTH: i32>(in %id: i32, in %data: !hw.int<#hw.param.decl.ref<"WIDTH">>, out out: i32) {
  hw.output %id : i32
}

// CHECK-LABEL: hw.module @UseWithExistingParams
hw.module @UseWithExistingParams(in %data: i16, out out0: i32, out out1: i32) {
  %c7_i32 = hw.constant 7 : i32

  // CHECK: %inst0.out = hw.instance "inst0" @WithExistingParams<WIDTH: i32 = 16, id: i32 = 7>
  // CHECK-SAME: (data: %data: i16) -> (out: i32)
  %inst0.out = hw.instance "inst0" @WithExistingParams<WIDTH: i32 = 16>(id: %c7_i32: i32, data: %data: i16) -> (out: i32)

  // CHECK: %inst1.out = hw.instance "inst1" @WithExistingParams<WIDTH: i32 = 16, id: i32 = 7>
  // CHECK-SAME: (data: %data: i16) -> (out: i32)
  %inst1.out = hw.instance "inst1" @WithExistingParams<WIDTH: i32 = 16>(id: %c7_i32: i32, data: %data: i16) -> (out: i32)

  hw.output %inst0.out, %inst1.out : i32, i32
}

// Test that ports with inner symbols are NOT parameterized, but other constant ports are
// CHECK-LABEL: hw.module private @WithInnerSym<data: i16>
// CHECK-SAME: (in %config : i16 {hw.exportPort = #hw<innerSym@sym>}, out out : i16)
// CHECK: hw.param.value i16 = #hw.param.decl.ref<"data">
// CHECK: hw.output
hw.module private @WithInnerSym(in %config: i16 {hw.exportPort = #hw<innerSym@sym>}, in %data: i16, out out: i16) {
  hw.output %data : i16
}

// CHECK-LABEL: hw.module @UseWithInnerSym
hw.module @UseWithInnerSym(in %data: i16, out out: i16) {
  %c42_i16 = hw.constant 42 : i16

  // CHECK: %inst.out = hw.instance "inst" @WithInnerSym<data: i16 = 42>
  // CHECK-SAME: (config: %c42_i16: i16) -> (out: i16)
  %inst.out = hw.instance "inst" @WithInnerSym(config: %c42_i16: i16, data: %c42_i16: i16) -> (out: i16)

  hw.output %inst.out : i16
}
