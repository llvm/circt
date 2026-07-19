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

// CHECK-LABEL: func.func private @test_null1() -> !llvm.ptr
// CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: return [[NULL]] : !llvm.ptr
func.func private @test_null1() -> !moore.null {
  %null = moore.null
  return %null : !moore.null
}

// CHECK-LABEL: func.func private @test_handle_eq1
// CHECK-SAME: (%arg0: !llvm.ptr) -> i1
// CHECK: [[NULL:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: [[EQ:%.+]] = llvm.icmp "eq" %arg0, [[NULL]] : !llvm.ptr
// CHECK: return [[EQ]] : i1
func.func private @test_handle_eq1(%arg0: !moore.class<@H>) -> !moore.i1 {
  %null = moore.null
  %eq = moore.handle_eq %arg0, %null : !moore.class<@H> : !moore.null -> i1
  return %eq : !moore.i1
}
moore.class.classdecl @H {
  moore.class.propertydecl @a : !moore.i32
}

// CHECK-LABEL: func.func private @test_handle_ne1
// CHECK-SAME: (%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i1
// CHECK: [[NE:%.+]] = llvm.icmp "ne" %arg0, %arg1 : !llvm.ptr
// CHECK: return [[NE]] : i1
func.func private @test_handle_ne1(%arg0: !moore.chandle, %arg1: !moore.chandle) -> !moore.i1 {
  %ne = moore.handle_ne %arg0, %arg1 : !moore.chandle : !moore.chandle -> i1
  return %ne : !moore.i1
}

// CHECK-LABEL: func.func private @test_null_eq_null
// CHECK: [[N0:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: [[N1:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK: [[EQ:%.+]] = llvm.icmp "eq" [[N0]], [[N1]] : !llvm.ptr
// CHECK: return [[EQ]] : i1
func.func private @test_null_eq_null() -> !moore.i1 {
  %n0 = moore.null
  %n1 = moore.null
  %eq = moore.handle_eq %n0, %n1 : !moore.null : !moore.null -> i1
  return %eq : !moore.i1
}
