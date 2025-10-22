// RUN: circt-verilog %s --parse-only | FileCheck %s

/// Flag tests

// CHECK-LABEL: moore.class.classdecl  Concrete @plain : {
// CHECK: }
class plain;
endclass

// CHECK-LABEL: moore.class.classdecl  Abstract @abstractOnly : {
// CHECK: }
virtual class abstractOnly;
endclass

// CHECK-LABEL: moore.class.classdecl  Interface @interfaceTestClass : {
// CHECK: }
interface class interfaceTestClass;
endclass

/// Interface tests

// CHECK-LABEL: moore.class.classdecl  Concrete @interfaceTestClass2 implements [@interfaceTestClass] : {
// CHECK: }
class interfaceTestClass2 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl  Interface @interfaceTestClass3 implements [@interfaceTestClass] : {
// CHECK: }
interface class interfaceTestClass3 extends interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl  Concrete @interfaceTestClass4 implements [@interfaceTestClass3] : {
// CHECK: }
class interfaceTestClass4 implements interfaceTestClass3;
endclass

/// Inheritance tests

// CHECK-LABEL: moore.class.classdecl  Concrete @inheritanceTest : {
// CHECK: }
class inheritanceTest;
endclass

// CHECK-LABEL: moore.class.classdecl  Concrete @inheritanceTest2 extends @inheritanceTest : {
// CHECK: }
class inheritanceTest2 extends inheritanceTest;
endclass

// Inheritance + interface tests

// CHECK-LABEL: moore.class.classdecl  Concrete @D extends @plain : {
// CHECK: }
class D extends plain;
endclass

// CHECK-LABEL: moore.class.classdecl  Concrete @Impl1 implements [@interfaceTestClass] : {
// CHECK: }
class Impl1 implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl  Concrete @Impl2 implements [@interfaceTestClass, @interfaceTestClass3] : {
// CHECK: }
class Impl2 implements interfaceTestClass, interfaceTestClass3;
endclass

// CHECK-LABEL: moore.class.classdecl  Concrete @DI extends @D implements [@interfaceTestClass] : {
// CHECK: }
class DI extends D implements interfaceTestClass;
endclass

// CHECK-LABEL: moore.class.classdecl  Interface @IMulti implements [@interfaceTestClass, @interfaceTestClass3] : {
// CHECK: }
interface class IMulti extends interfaceTestClass, interfaceTestClass3;
endclass

/// Property tests

// CHECK-LABEL: moore.class.classdecl  Concrete @PropertyCombo : {
// CHECK:   moore.class.propertydecl[ Public,  Automatic] @pubAutoI32 : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl[ Protected,  Static] @protStatL18 : !moore.l18
// CHECK-NEXT:   moore.class.propertydecl[ Local,  Automatic] @localAutoI32 : !moore.i32
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
// CHECK-LABEL: moore.class.classdecl  Concrete @PropertyOrder : {
// CHECK:   moore.class.propertydecl[ Public,  Automatic] @a : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl[ Public,  Automatic] @b : !moore.i32
// CHECK-NEXT:   moore.class.propertydecl[ Public,  Automatic] @c : !moore.i32
// CHECK: }
class PropertyOrder;
  int a;
  int b;
  int c;
endclass

// Classes within packages
package testPackage;
   // CHECK-LABEL: moore.class.classdecl  Concrete @"testPackage::testPackageClass" : {
   class testPackageClass;
   // CHECK: }
   endclass
endpackage

// CHECK-LABEL: moore.module @testModule() {
// CHECK: }
// CHECK: moore.class.classdecl  Concrete @"testModule::testModuleClass" : {
// CHECK: }
module testModule #();
   class testModuleClass;
   endclass
endmodule

// CHECK-LABEL: moore.class.classdecl  Concrete @testClass : {
// CHECK: }
// CHECK: moore.class.classdecl  Concrete @"testClass::testClass" : {
// CHECK: }
class testClass;
   class testClass;
   endclass // testClass
endclass

// CHECK-LABEL:  moore.module @testModule2() {
module testModule2 #();
    class testModuleClass;
    endclass // testModuleClass2
    // CHECK-NEXT: [[OBJ:%.+]] = moore.variable : <class.object<@"testModule2::testModuleClass">>
    testModuleClass t;
    // CHECK-NEXT:     moore.output
    // CHECK-NEXT:   }
endmodule
// CHECK: moore.class.classdecl  Concrete @"testModule2::testModuleClass" : {
// CHECK: }

// CHECK-LABEL: moore.module @testModule3() {
module testModule3;
    class testModuleClass;
    endclass
    // CHECK: [[T:%.*]] = moore.variable : <class.object<@"testModule3::testModuleClass">>
    testModuleClass t;
    // CHECK: moore.procedure initial {
    initial begin
        // CHECK:   [[NEW:%.*]] = moore.class.new : <@"testModule3::testModuleClass">
        // CHECK:   moore.blocking_assign [[T]], [[NEW]] : class.object<@"testModule3::testModuleClass">
        t = new;
        // CHECK:   moore.return
        // CHECK: }
    end
    // CHECK: moore.output
endmodule
