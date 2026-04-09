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
// CHECK-DAG: llvm.mlir.global internal constant @"tClass::vtable"()
// CHECK-DAG: llvm.mlir.addressof @"tClass::subroutine" : !llvm.ptr
// CHECK-DAG: llvm.mlir.addressof @"testClass::testSubroutine" : !llvm.ptr
// CHECK-DAG: llvm.mlir.global internal constant @"testClass::vtable"()
// CHECK-DAG: llvm.mlir.addressof @"testClass::subroutine" : !llvm.ptr
// CHECK-DAG: llvm.mlir.addressof @"testClass::testSubroutine" : !llvm.ptr

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
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"C::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[HEADERPTR:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>
// CHECK:   [[TYPEINFOPTR:%.*]] = llvm.getelementptr [[HEADERPTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:   llvm.store [[TYPEINFO]], [[TYPEINFOPTR]] : !llvm.ptr, !llvm.ptr
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
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"D::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[HEADERPTR:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"D", (struct<(ptr, ptr)>, struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>, i32, i64, i16)>
// CHECK:   [[TYPEINFOPTR:%.*]] = llvm.getelementptr [[HEADERPTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:   llvm.store [[TYPEINFO]], [[TYPEINFOPTR]] : !llvm.ptr, !llvm.ptr
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
// CHECK-SAME: (%arg0: !llvm.ptr) -> !llvm.ptr {
// CHECK:   [[CONSTIDX:%.+]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:   [[GEP:%.+]] = llvm.getelementptr %arg0[[[CONSTIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"G", (struct<(ptr, ptr)>, struct<"C", (struct<(ptr, ptr)>, i32, i32, i32)>, i32, i32, i32)>
// CHECK:   return [[GEP]] : !llvm.ptr

// CHECK-NOT: moore.class.property_ref
// CHECK-NOT: moore.class.classdecl

func.func private @test_new6(%arg0: !moore.class<@G>) -> !moore.ptr<i32> {
  %gep = moore.class.property_ref %arg0[@d] : <@G> -> !moore.ptr<i32>
  return %gep : !moore.ptr<i32>
}
moore.class.classdecl @G extends @C {
  moore.class.propertydecl @d : !moore.i32
  moore.class.propertydecl @e : !moore.l32
  moore.class.propertydecl @f : !moore.l32
}

/// Check that virtual classes use the same object header layout.

// CHECK-LABEL: func.func private @test_new7
// CHECK:   [[SIZE:%.*]] = llvm.mlir.constant(24 : i64) : i64
// CHECK:   [[PTR:%.*]] = llvm.call @malloc([[SIZE]]) : (i64) -> !llvm.ptr
// CHECK:   [[TYPEINFO:%.*]] = llvm.mlir.addressof @"VirtualC::typeinfo" : !llvm.ptr
// CHECK:   [[HEADERIDX:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:   [[HEADERPTR:%.*]] = llvm.getelementptr [[PTR]][[[HEADERIDX]]] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"VirtualC", (struct<(ptr, ptr)>, i32)>
// CHECK:   [[TYPEINFOPTR:%.*]] = llvm.getelementptr [[HEADERPTR]][[[HEADERIDX]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(ptr, ptr)>
// CHECK:   llvm.store [[TYPEINFO]], [[TYPEINFOPTR]] : !llvm.ptr, !llvm.ptr
// CHECK:   return

func.func private @test_new7() {
  %h = moore.class.new : <@VirtualC>
  return
}
moore.class.classdecl @VirtualC {
  moore.class.propertydecl @a : !moore.i32
  moore.class.methoddecl @f : (!moore.class<@VirtualC>) -> ()
}

/// Check that symbolic vtables lower to LLVM globals and convert only the
/// referenced methods to llvm.func.

// CHECK-LABEL: llvm.func @"testClass::subroutine"(
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @"testClass::testSubroutine"(
// CHECK: llvm.return

// CHECK-LABEL: llvm.func @"tClass::subroutine"(
// CHECK: llvm.return

// CHECK-NOT: moore.vtable
// CHECK-NOT: moore.vtable_entry

moore.class.classdecl @virtualFunctionClass {
  moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>) -> ()
}
moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
  moore.class.methoddecl @testSubroutine : (!moore.class<@realFunctionClass>) -> ()
}
moore.class.classdecl @testClass implements [@realFunctionClass] {
  moore.class.methoddecl @subroutine -> @"testClass::subroutine" : (!moore.class<@testClass>) -> ()
  moore.class.methoddecl @testSubroutine -> @"testClass::testSubroutine" : (!moore.class<@testClass>) -> ()
}
moore.vtable @testClass::@vtable {
  moore.vtable @realFunctionClass::@vtable {
    moore.vtable @virtualFunctionClass::@vtable {
      moore.vtable_entry @subroutine -> @"testClass::subroutine"
    }
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  moore.vtable_entry @subroutine -> @"testClass::subroutine"
  moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
}
func.func private @"testClass::subroutine"(%arg0: !moore.class<@testClass>) {
  return
}
func.func private @"testClass::testSubroutine"(%arg0: !moore.class<@testClass>) {
  return
}

moore.class.classdecl @tClass extends @testClass {
  moore.class.methoddecl @subroutine -> @"tClass::subroutine" : (!moore.class<@tClass>) -> ()
}
moore.vtable @tClass::@vtable {
  moore.vtable @testClass::@vtable {
    moore.vtable @realFunctionClass::@vtable {
      moore.vtable @virtualFunctionClass::@vtable {
        moore.vtable_entry @subroutine -> @"tClass::subroutine"
      }
      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
    }
    moore.vtable_entry @subroutine -> @"tClass::subroutine"
    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
  }
  moore.vtable_entry @subroutine -> @"tClass::subroutine"
}
func.func private @"tClass::subroutine"(%arg0: !moore.class<@tClass>) {
  return
}
