// RUN: circt-verilog %s --parse-only | FileCheck %s

/// Flag tests

// CHECK-LABEL: moore.class.classdecl @plain {
// CHECK: }
class plain;
endclass

// CHECK-LABEL: moore.class.classdecl @abstractOnly {
// CHECK: }
virtual class abstractOnly;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass {
// CHECK: }
interface class interfaceTestClass;
endclass

/// Interface tests

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass2 implements [@interfaceTestClass] {
// CHECK: }
class interfaceTestClass2 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass3 implements [@interfaceTestClass] {
// CHECK: }
interface class interfaceTestClass3 extends interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @interfaceTestClass4 implements [@interfaceTestClass3] {
// CHECK: }
class interfaceTestClass4 implements interfaceTestClass3;
endclass

/// Inheritance tests

// CHECK-LABEL: moore.class.classdecl @inheritanceTest {
// CHECK: }
class inheritanceTest;
endclass

// CHECK-LABEL: moore.class.classdecl @inheritanceTest2 extends @inheritanceTest {
// CHECK: }
class inheritanceTest2 extends inheritanceTest;
endclass

// Inheritance + interface tests

// CHECK-LABEL: moore.class.classdecl @D extends @plain {
// CHECK: }
class D extends plain;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl1 implements [@interfaceTestClass] {
// CHECK: }
class Impl1 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @Impl2 implements [@interfaceTestClass, @interfaceTestClass3] {
// CHECK: }
class Impl2 implements interfaceTestClass, interfaceTestClass3;
endclass

// CHECK-LABEL: moore.class.classdecl @DI extends @D implements [@interfaceTestClass] {
// CHECK: }
class DI extends D implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl @IMulti implements [@interfaceTestClass, @interfaceTestClass3] {
// CHECK: }
interface class IMulti extends interfaceTestClass, interfaceTestClass3;
endclass

/// Property tests

// CHECK-LABEL: moore.class.classdecl @PropertyCombo {
// CHECK:   moore.class.propertydecl @pubAutoI32 : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @protStatL18 : !moore.l18
// CHECK-NEXT:   moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK: }
class PropertyCombo;
  // public automatic int
  int pubAutoI32;

  // protected static logic [17:0]
  protected static logic [17:0] protStatL18;

  // local automatic int
  local int localAutoI32;
endclass

// Ensure multiple propertys preserve declaration order
// CHECK-LABEL: moore.class.classdecl @PropertyOrder {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @b : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl @c : !moore.i32
// CHECK: }
class PropertyOrder;
  int a;
  int b;
  int c;
endclass

// Classes within packages
package testPackage;
   // CHECK-LABEL: moore.class.classdecl @"testPackage::testPackageClass" {
   class testPackageClass;
   // CHECK: }
   endclass
endpackage

// CHECK-LABEL: moore.module @testModule() {
// CHECK: }
// CHECK: moore.class.classdecl @"testModule::testModuleClass" {
// CHECK: }
module testModule #();
   class testModuleClass;
   endclass
endmodule

// CHECK-LABEL: moore.class.classdecl @testClass {
// CHECK: }
// CHECK: moore.class.classdecl @"testClass::testClass" {
// CHECK: }
class testClass;
   class testClass;
   endclass // testClass
endclass

/// Check handle variable

// CHECK-LABEL:  moore.module @testModule2() {
// CHECK-NEXT: [[OBJ:%.+]] = moore.variable : <class<@"testModule2::testModuleClass">>
// CHECK-NEXT:     moore.output
// CHECK-NEXT:   }
// CHECK: moore.class.classdecl @"testModule2::testModuleClass" {
// CHECK: }
module testModule2 #();
    class testModuleClass;
    endclass // testModuleClass2
    testModuleClass t;

endmodule

/// Check calls to new without explicit constructor

// CHECK-LABEL: moore.module @testModule3() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule3::testModuleClass">>
// CHECK: moore.procedure initial {
// CHECK:   [[NEW:%.*]] = moore.class.new : <@"testModule3::testModuleClass">
// CHECK:   moore.blocking_assign [[T]], [[NEW]] : class<@"testModule3::testModuleClass">
// CHECK:   moore.return
// CHECK: }
// CHECK: moore.output

module testModule3;
    class testModuleClass;
    endclass
    testModuleClass t;
    initial begin
        t = new;
    end
endmodule

/// Check property read access

// CHECK-LABEL: moore.module @testModule4() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule4::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule4::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule4::testModuleClass">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule4::testModuleClass">>
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[CLASSHANDLE]][@a] : <@"testModule4::testModuleClass"> -> <i32>
// CHECK:    [[A:%.+]] = moore.read [[REF]]
// CHECK:    moore.blocking_assign [[RESULT]], [[A]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule4::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule4;
    class testModuleClass;
       int a;
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        result = t.a;
    end
endmodule

/// Check property write access

// CHECK-LABEL: moore.module @testModule5() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule5::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule5::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule5::testModuleClass">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule5::testModuleClass">>
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[CLASSHANDLE]][@a] : <@"testModule5::testModuleClass"> -> <i32>
// CHECK:    [[RESR:%.+]] = moore.read [[RESULT]] : <i32>
// CHECK:    moore.blocking_assign [[REF]], [[RESR]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule5::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule5;
    class testModuleClass;
       int a;
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        t.a = result;
    end
endmodule

/// Check implicit upcast

// CHECK-LABEL: moore.module @testModule6() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule6::testModuleClass2">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule6::testModuleClass2">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule6::testModuleClass2">
// CHECK:    [[CLASSHANDLE:%.+]] = moore.read [[T]] : <class<@"testModule6::testModuleClass2">>
// CHECK:    [[UPCAST:%.+]] = moore.class.upcast [[CLASSHANDLE]] : <@"testModule6::testModuleClass2"> to <@"testModule6::testModuleClass">
// CHECK:    [[REF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <@"testModule6::testModuleClass"> -> <i32>
// CHECK:    [[A:%.+]] = moore.read [[REF]]
// CHECK:    moore.blocking_assign [[RESULT]], [[A]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule6::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK: }

module testModule6;
    class testModuleClass;
       int a;
    endclass
    class testModuleClass2 extends testModuleClass;
    endclass
    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.a;
    end
endmodule
