// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

// CHECK-DAG: llvm.mlir.global internal constant @"C::typeinfo"() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
// CHECK-DAG: llvm.mlir.zero : !llvm.ptr
// CHECK-DAG: llvm.insertvalue
// CHECK-DAG: llvm.mlir.global internal constant @"D::typeinfo"() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
// CHECK-DAG: llvm.mlir.addressof @"C::typeinfo" : !llvm.ptr
// CHECK-DAG: llvm.insertvalue
// CHECK-DAG: llvm.mlir.global internal constant @"VirtualC::typeinfo"() {addr_space = 0 : i32} : !llvm.struct<(ptr)> {
// CHECK-DAG: llvm.mlir.zero : !llvm.ptr
// CHECK-DAG: llvm.insertvalue

/// Check that a classdecl gets noop'd and handles are lowered to !llvm.ptr

// CHECK-LABEL:   func.func @ClassType(%arg0: !llvm.ptr) {
// CHECK:    return
// CHECK:  }
// CHECK-NOT: moore.class.classdecl
// CHECK-NOT: moore.class<@PropertyCombo>

moore.class.classdecl @PropertyCombo {
  moore.class.propertydecl @pubAutoI32   : !moore.i32
  moore.class.propertydecl @protStatL18  : !moore.l18
  moore.class.propertydecl @localAutoI32 : !moore.i32
}

func.func @ClassType(%arg0: !moore.class<@PropertyCombo>) {
  return
}

/// Check that new lowers to malloc

// CHECK-LABEL: func.func private @test_new2
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"C::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[GEP:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>
// CHECK:   llvm.store [[TYPEINFO]], [[GEP]] : !llvm.ptr, !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

// Allocate a new instance; should lower to call @malloc(i64).
func.func private @test_new2() {
  %h = moore.class.new : <@C>
  return
}
// Minimal class so the identified struct has a concrete body.
moore.class.classdecl @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that new lowers to malloc with inheritance without shadowing

// CHECK-LABEL: func.func private @test_new3
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"D::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[GEP:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"D", (struct<(ptr, ptr)>, struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>, i32, i64, i16)>
// CHECK:   llvm.store [[TYPEINFO]], [[GEP]] : !llvm.ptr, !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_new3() {
  %h = moore.class.new : <@D>
  return
}
moore.class.classdecl @D extends @C {
  moore.class.propertydecl @d : !moore.l32
  moore.class.propertydecl @e : !moore.l64
  moore.class.propertydecl @f : !moore.i16
}

/// Check that new lowers to malloc with inheritance & shadowing

// CHECK-LABEL: func.func private @test_new4
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(64 : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

func.func private @test_new4() {
  %h = moore.class.new : <@E>
  return
}
moore.class.classdecl @E extends @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that upcast lowers to no-op

// CHECK-LABEL: func.func private @test_new5
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK:   return %arg0 : !llvm.ptr

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.upcast
// CHECK-NOT: moore.class.classdecl

func.func private @test_new5(%arg0: !moore.class<@F>) -> !moore.class<@C> {
  %upcast = moore.class.upcast %arg0 : <@F> to <@C>
  return %upcast : !moore.class<@C>
}
moore.class.classdecl @F extends @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that property_ref lowers to GEP

// CHECK-LABEL: func.func private @test_new6
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<i32> {
// CHECK:   [[CONSTIDX:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[[[CONSTIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"G", (struct<(ptr, ptr)>, struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>, i32, i32, i32)>
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return [[CONV]] : !llhd.ref<i32>

// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.classdecl

func.func private @test_new6(%arg0: !moore.class<@G>) -> !moore.ref<i32> {
  %gep = moore.class.property_ref %arg0[@d] : <@G> -> !moore.ref<i32>
  return %gep : !moore.ref<i32>
}
moore.class.classdecl @G extends @C {
  moore.class.propertydecl @d : !moore.i32
  moore.class.propertydecl @e : !moore.l32
  moore.class.propertydecl @f : !moore.l32
}

/// Check that virtual classes use the same object header layout.

// CHECK-LABEL: func.func private @test_new7
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"VirtualC::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[GEP:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"VirtualC", (struct<(ptr, ptr)>, i32)>
// CHECK:   llvm.store [[TYPEINFO]], [[GEP]] : !llvm.ptr, !llvm.ptr
// CHECK:   return

func.func private @test_new7() {
  %h = moore.class.new : <@VirtualC>
  return
}
moore.class.classdecl @VirtualC {
  moore.class.propertydecl @a : !moore.i32
  moore.class.methoddecl @f : (!moore.class<@VirtualC>) -> ()
}

/// Check that class properties with aggregate Moore value types lower to
/// LLVM-compatible class object storage types.

// CHECK-LABEL: func.func private @test_new_aggregate_properties
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant({{[0-9]+}} : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   llvm.getelementptr [[PTR]]{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"AggregateProperties", (struct<(ptr, ptr)>, array<2 x ptr>, struct<(ptr, f64, i8)>, array<8 x i8>)>
// CHECK:   return

func.func private @test_new_aggregate_properties() {
  %h = moore.class.new : <@AggregateProperties>
  return
}

// CHECK-LABEL: func.func private @test_ref_aggregate_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<!hw.array<2x!llvm.ptr>> {
// CHECK:   llvm.getelementptr %arg0{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"AggregateProperties", (struct<(ptr, ptr)>, array<2 x ptr>, struct<(ptr, f64, i8)>, array<8 x i8>)>
// CHECK:   builtin.unrealized_conversion_cast {{.*}} : !llvm.ptr to !llhd.ref<!hw.array<2x!llvm.ptr>>
// CHECK:   return

func.func private @test_ref_aggregate_property(%arg0: !moore.class<@AggregateProperties>) -> !moore.ref<uarray<2 x chandle>> {
  %gep = moore.class.property_ref %arg0[@handles] : <@AggregateProperties> -> !moore.ref<uarray<2 x chandle>>
  return %gep : !moore.ref<uarray<2 x chandle>>
}

moore.class.classdecl @AggregateProperties {
  moore.class.propertydecl @handles : !moore.uarray<2 x chandle>
  moore.class.propertydecl @payload : !moore.ustruct<{h: chandle, r: f64, b: i8}>
  moore.class.propertydecl @choice : !moore.uunion<{h: chandle, r: f64, b: i8}>
}

/// Check that time properties and aggregate time leaves use LLVM-compatible
/// storage while refs still preserve the LLHD time element type.

// CHECK-LABEL: func.func private @test_new_time_aggregate_properties
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant({{[0-9]+}} : i64) : i64
// CHECK:   [[PTR:%.*]] = call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   llvm.getelementptr [[PTR]]{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"TimeAggregateProperties", (struct<(ptr, ptr)>, i64, array<2 x i64>, struct<(i64, i8)>, array<8 x i8>)>
// CHECK:   return

func.func private @test_new_time_aggregate_properties() {
  %h = moore.class.new : <@TimeAggregateProperties>
  return
}

// CHECK-LABEL: func.func private @test_read_time_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.time {
// CHECK:   [[GEP:%.*]] = llvm.getelementptr %arg0{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"TimeAggregateProperties", (struct<(ptr, ptr)>, i64, array<2 x i64>, struct<(i64, i8)>, array<8 x i8>)>
// CHECK:   [[REF:%.*]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<!llhd.time>
// CHECK:   [[VALUE:%.*]] = llhd.prb [[REF]] : !llhd.time
// CHECK:   return [[VALUE]] : !llhd.time

func.func private @test_read_time_property(%arg0: !moore.class<@TimeAggregateProperties>) -> !moore.time {
  %ref = moore.class.property_ref %arg0[@stamp] : <@TimeAggregateProperties> -> !moore.ref<time>
  %value = moore.read %ref : <!moore.time>
  return %value : !moore.time
}

// CHECK-LABEL: func.func private @test_ref_time_array_property
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llhd.ref<!hw.array<2x!llhd.time>> {
// CHECK:   llvm.getelementptr %arg0{{.*}} : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"TimeAggregateProperties", (struct<(ptr, ptr)>, i64, array<2 x i64>, struct<(i64, i8)>, array<8 x i8>)>
// CHECK:   builtin.unrealized_conversion_cast {{.*}} : !llvm.ptr to !llhd.ref<!hw.array<2x!llhd.time>>
// CHECK:   return

func.func private @test_ref_time_array_property(%arg0: !moore.class<@TimeAggregateProperties>) -> !moore.ref<uarray<2 x time>> {
  %ref = moore.class.property_ref %arg0[@stamps] : <@TimeAggregateProperties> -> !moore.ref<uarray<2 x time>>
  return %ref : !moore.ref<uarray<2 x time>>
}

moore.class.classdecl @TimeAggregateProperties {
  moore.class.propertydecl @stamp : !moore.time
  moore.class.propertydecl @stamps : !moore.uarray<2 x time>
  moore.class.propertydecl @payload : !moore.ustruct<{stamp: time, bits: i8}>
  moore.class.propertydecl @choice : !moore.uunion<{stamp: time, bits: i64}>
}
