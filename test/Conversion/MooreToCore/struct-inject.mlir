// RUN: circt-opt %s --convert-moore-to-core | FileCheck %s

// CHECK-LABEL: func.func @StructInject
// CHECK-SAME: (%arg0: !hw.struct<a: i3, b: i2>, %arg1: i3) -> !hw.struct<a: i3, b: i2>
func.func @StructInject(%arg0: !moore.struct<{a: i3, b: i2}>, %arg1: !moore.i3) -> !moore.struct<{a: i3, b: i2}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["a"], %arg1 : !hw.struct<a: i3, b: i2>
  // CHECK: return [[RESULT]] : !hw.struct<a: i3, b: i2>
  %0 = moore.struct_inject %arg0, "a", %arg1 : struct<{a: i3, b: i2}>, i3
  return %0 : !moore.struct<{a: i3, b: i2}>
}

// CHECK-LABEL: func.func @UnpackedStructInject
// CHECK-SAME: (%arg0: !hw.struct<a: i3, b: i2>, %arg1: i2) -> !hw.struct<a: i3, b: i2>
func.func @UnpackedStructInject(%arg0: !moore.ustruct<{a: i3, b: i2}>, %arg1: !moore.i2) -> !moore.ustruct<{a: i3, b: i2}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["b"], %arg1 : !hw.struct<a: i3, b: i2>
  // CHECK: return [[RESULT]] : !hw.struct<a: i3, b: i2>
  %0 = moore.struct_inject %arg0, "b", %arg1 : ustruct<{a: i3, b: i2}>, i2
  return %0 : !moore.ustruct<{a: i3, b: i2}>
}

// CHECK-LABEL: func.func @UnpackedStructInjectString
// CHECK-SAME: (%arg0: !hw.struct<s: !sim.dstring, h: !llvm.ptr>, %arg1: !sim.dstring)
// CHECK-SAME: -> !hw.struct<s: !sim.dstring, h: !llvm.ptr>
func.func @UnpackedStructInjectString(%arg0: !moore.ustruct<{s: string, h: chandle}>, %arg1: !moore.string) -> !moore.ustruct<{s: string, h: chandle}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["s"], %arg1 : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  // CHECK: return [[RESULT]] : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  %0 = moore.struct_inject %arg0, "s", %arg1 : ustruct<{s: string, h: chandle}>, string
  return %0 : !moore.ustruct<{s: string, h: chandle}>
}

// CHECK-LABEL: func.func @UnpackedStructInjectChandle
// CHECK-SAME: (%arg0: !hw.struct<s: !sim.dstring, h: !llvm.ptr>, %arg1: !llvm.ptr)
// CHECK-SAME: -> !hw.struct<s: !sim.dstring, h: !llvm.ptr>
func.func @UnpackedStructInjectChandle(%arg0: !moore.ustruct<{s: string, h: chandle}>, %arg1: !moore.chandle) -> !moore.ustruct<{s: string, h: chandle}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["h"], %arg1 : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  // CHECK: return [[RESULT]] : !hw.struct<s: !sim.dstring, h: !llvm.ptr>
  %0 = moore.struct_inject %arg0, "h", %arg1 : ustruct<{s: string, h: chandle}>, chandle
  return %0 : !moore.ustruct<{s: string, h: chandle}>
}

moore.class.classdecl @StructInjectClass {
}

// CHECK-LABEL: func.func @UnpackedStructInjectClass
// CHECK-SAME: (%arg0: !hw.struct<s: !sim.dstring, c: !llvm.ptr>, %arg1: !llvm.ptr)
// CHECK-SAME: -> !hw.struct<s: !sim.dstring, c: !llvm.ptr>
func.func @UnpackedStructInjectClass(%arg0: !moore.ustruct<{s: string, c: class<@StructInjectClass>}>, %arg1: !moore.class<@StructInjectClass>) -> !moore.ustruct<{s: string, c: class<@StructInjectClass>}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["c"], %arg1 : !hw.struct<s: !sim.dstring, c: !llvm.ptr>
  // CHECK: return [[RESULT]] : !hw.struct<s: !sim.dstring, c: !llvm.ptr>
  %0 = moore.struct_inject %arg0, "c", %arg1 : ustruct<{s: string, c: class<@StructInjectClass>}>, class<@StructInjectClass>
  return %0 : !moore.ustruct<{s: string, c: class<@StructInjectClass>}>
}

// CHECK-LABEL: func.func @UnpackedStructInjectReal
// CHECK-SAME: (%arg0: !hw.struct<r: f64, t: !llhd.time>, %arg1: f64)
// CHECK-SAME: -> !hw.struct<r: f64, t: !llhd.time>
func.func @UnpackedStructInjectReal(%arg0: !moore.ustruct<{r: f64, t: time}>, %arg1: !moore.f64) -> !moore.ustruct<{r: f64, t: time}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["r"], %arg1 : !hw.struct<r: f64, t: !llhd.time>
  // CHECK: return [[RESULT]] : !hw.struct<r: f64, t: !llhd.time>
  %0 = moore.struct_inject %arg0, "r", %arg1 : ustruct<{r: f64, t: time}>, f64
  return %0 : !moore.ustruct<{r: f64, t: time}>
}

// CHECK-LABEL: func.func @UnpackedStructInjectTime
// CHECK-SAME: (%arg0: !hw.struct<r: f64, t: !llhd.time>, %arg1: !llhd.time)
// CHECK-SAME: -> !hw.struct<r: f64, t: !llhd.time>
func.func @UnpackedStructInjectTime(%arg0: !moore.ustruct<{r: f64, t: time}>, %arg1: !moore.time) -> !moore.ustruct<{r: f64, t: time}> {
  // CHECK: [[RESULT:%.+]] = hw.struct_inject %arg0["t"], %arg1 : !hw.struct<r: f64, t: !llhd.time>
  // CHECK: return [[RESULT]] : !hw.struct<r: f64, t: !llhd.time>
  %0 = moore.struct_inject %arg0, "t", %arg1 : ustruct<{r: f64, t: time}>, time
  return %0 : !moore.ustruct<{r: f64, t: time}>
}
