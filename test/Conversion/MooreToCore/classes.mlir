// RUN: circt-opt %s --convert-moore-to-core --verify-diagnostics | FileCheck %s

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

// malloc should be declared in the LLVM dialect.
// CHECK-LABEL: func.func private @test_new2
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(12 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.classdecl

// Allocate a new instance; should lower to llvm.call @malloc(i64).
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
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(28 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
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
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
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
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.upcast
// CHECK-NOT: moore.class.classdecl

func.func private @test_new5() {
  %h = moore.class.new : <@F>
  %upcast = moore.class.upcast %h : <@F> to <@C>
  return
}
moore.class.classdecl @F extends @C {
  moore.class.propertydecl @a : !moore.i32
  moore.class.propertydecl @b : !moore.l32
  moore.class.propertydecl @c : !moore.l32
}

/// Check that property_ref lowers to GEP

// CHECK-LABEL: func.func private @test_new6
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[CONSTIDX:%.+]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:   [[GEP:%.+]] = llvm.getelementptr [[PTR]][[[CONSTIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"G", (struct<"C", (i32, i32, i32)>, i32, i32, i32)>
// CHECK:   [[CONV:%.+]] = builtin.unrealized_conversion_cast [[GEP]] : !llvm.ptr to !llhd.ref<i32>
// CHECK:   return

// CHECK-NOT: moore.class.new
// CHECK-NOT: moore.class.upcast
// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.classdecl

func.func private @test_new6() {
  %h = moore.class.new : <@G>
  %gep = moore.class.property_ref %h[@d] : <@G> -> <i32>
  return
}
moore.class.classdecl @G extends @C {
  moore.class.propertydecl @d : !moore.i32
  moore.class.propertydecl @e : !moore.l32
  moore.class.propertydecl @f : !moore.l32
}
