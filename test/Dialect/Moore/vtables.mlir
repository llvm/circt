// RUN: circt-opt %s --moore-create-vtables --verify-diagnostics | FileCheck %s

// CHECK-LABEL:  moore.vtable @testClass::@vtable {
// CHECK-NEXT:    moore.vtable @realFunctionClass::@vtable {
// CHECK-NEXT:      moore.vtable @virtualFunctionClass::@vtable {
// CHECK-NEXT:        moore.vtable_entry @subroutine -> @"testClass::subroutine"
// CHECK-NEXT:      }
// CHECK-NEXT:      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK-NEXT:    }
// CHECK-NEXT:    moore.vtable_entry @subroutine -> @"testClass::subroutine"
// CHECK-NEXT:    moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK-NEXT:  }

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
  func.func private @"testClass::subroutine"(%arg0: !moore.class<@testClass>) {
    return
  }
  func.func private @"testClass::testSubroutine"(%arg0: !moore.class<@testClass>) {
    return
  }

// CHECK-LABEL:  moore.vtable @tClass::@vtable {
// CHECK-NEXT:    moore.vtable @testClass::@vtable {
// CHECK-NEXT:      moore.vtable @realFunctionClass::@vtable {
// CHECK-NEXT:        moore.vtable @virtualFunctionClass::@vtable {
// CHECK-NEXT:          moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK-NEXT:        }
// CHECK-NEXT:        moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK-NEXT:      }
// CHECK-NEXT:      moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK-NEXT:      moore.vtable_entry @testSubroutine -> @"testClass::testSubroutine"
// CHECK-NEXT:    }
// CHECK-NEXT:    moore.vtable_entry @subroutine -> @"tClass::subroutine"
// CHECK-NEXT:  }

  moore.class.classdecl @tClass extends @testClass {
    moore.class.methoddecl @subroutine -> @"tClass::subroutine" : (!moore.class<@tClass>) -> ()
  }
  func.func private @"tClass::subroutine"(%arg0: !moore.class<@tClass>) {
    return
  }
