// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang

// Internal issue in Slang v3 about jump depending on uninitialised value.
// UNSUPPORTED: valgrind

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

/// Check concrete method calls

// CHECK-LABEL: moore.module @testModule7() {
// CHECK: [[T:%.*]] = moore.variable : <class<@"testModule7::testModuleClass">>
// CHECK: [[RESULT:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:    [[NEW:%.*]] = moore.class.new : <@"testModule7::testModuleClass">
// CHECK:    moore.blocking_assign [[T]], [[NEW]] : class<@"testModule7::testModuleClass">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[T]] : <class<@"testModule7::testModuleClass">>
// CHECK:    [[FUNCRET:%.+]] = func.call @"testModule7::testModuleClass::returnA"([[CALLREAD]]) : (!moore.class<@"testModule7::testModuleClass">) -> !moore.i32
// CHECK:    moore.blocking_assign [[RESULT]], [[FUNCRET]] : i32
// CHECK:    moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }

// CHECK: moore.class.classdecl @"testModule7::testModuleClass" {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.i32
// CHECK-NEXT: moore.class.methoddecl @returnA : (!moore.class<@"testModule7::testModuleClass">) -> !moore.i32
// CHECK: }

// CHECK: func.func private @"testModule7::testModuleClass::returnA"
// CHECK-SAME: ([[ARG:%.+]]: !moore.class<@"testModule7::testModuleClass">)
// CHECK-SAME: -> !moore.i32 {
// CHECK-NEXT: [[REF:%.+]] = moore.class.property_ref [[ARG]][@a] : <@"testModule7::testModuleClass"> -> <i32>
// CHECK-NEXT: [[RETURN:%.+]] = moore.read [[REF]] : <i32>
// CHECK-NEXT: return [[RETURN]] : !moore.i32
// CHECK-NEXT: }

module testModule7;
    class testModuleClass;
       int a;
       function int returnA();
          return a;
       endfunction
    endclass
    testModuleClass t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end
endmodule


/// Check inherited property access

 // CHECK-LABEL: moore.module @testModule8() {
 // CHECK:    [[t:%.+]] = moore.variable : <class<@"testModule8::testModuleClass2">>
 // CHECK:    [[result:%.+]] = moore.variable : <i32>
 // CHECK:    moore.procedure initial {
 // CHECK:      [[NEW:%.+]] = moore.class.new : <@"testModule8::testModuleClass2">
 // CHECK:      moore.blocking_assign [[t]], [[NEW]] : class<@"testModule8::testModuleClass2">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[t]] : <class<@"testModule8::testModuleClass2">>
// CHECK:       [[CALL:%.+]] = func.call @"testModule8::testModuleClass2::returnA"([[CALLREAD]]) : (!moore.class<@"testModule8::testModuleClass2">) -> !moore.i32
// CHECK:       moore.blocking_assign [[result]], [[CALL]] : i32
 // CHECK:      moore.return
 // CHECK:    }
 // CHECK:    moore.output
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule8::testModuleClass" {
 // CHECK:    moore.class.propertydecl @a : !moore.i32
 // CHECK:  }
 // CHECK:  moore.class.classdecl @"testModule8::testModuleClass2" extends @"testModule8::testModuleClass" {
 // CHECK:    moore.class.methoddecl @returnA : (!moore.class<@"testModule8::testModuleClass2">) -> !moore.i32
 // CHECK:  }
 // CHECK:  func.func private @"testModule8::testModuleClass2::returnA"([[ARG:%.+]]: !moore.class<@"testModule8::testModuleClass2">) -> !moore.i32 {
 // CHECK:   [[UPCAST:%.+]] = moore.class.upcast [[ARG]] : <@"testModule8::testModuleClass2"> to <@"testModule8::testModuleClass">
 // CHECK:   [[PROPREF:%.+]] = moore.class.property_ref [[UPCAST]][@a] : <@"testModule8::testModuleClass"> -> <i32>
 // CHECK:   [[RET:%.+]] = moore.read [[PROPREF]] : <i32>
 // CHECK:   return [[RET]] : !moore.i32
 // CHECK: }

module testModule8;

    class testModuleClass;
       int a;
    endclass // testModuleClass

   class testModuleClass2 extends testModuleClass;
       function int returnA();
          return a;
       endfunction
   endclass // testModuleClass2

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end

endmodule

/// Check method lowering without qualified handle

// CHECK-LABEL: moore.module @testModule9() {
// CHECK: [[t:%.+]] = moore.variable : <class<@"testModule9::testModuleClass2">>
// CHECK: [[result:%.+]] = moore.variable : <i32>
// CHECK: moore.procedure initial {
// CHECK:   [[new_obj:%.+]] = moore.class.new : <@"testModule9::testModuleClass2">
// CHECK:   moore.blocking_assign [[t]], [[new_obj]] : class<@"testModule9::testModuleClass2">
// CHECK:    [[CALLREAD:%.+]] = moore.read [[t]] : <class<@"testModule9::testModuleClass2">>
// CHECK:   [[call_ret:%.+]] = func.call @"testModule9::testModuleClass2::returnA"([[CALLREAD]]) : (!moore.class<@"testModule9::testModuleClass2">) -> !moore.i32
// CHECK:   moore.blocking_assign [[result]], [[call_ret]] : i32
// CHECK:   moore.return
// CHECK: }
// CHECK: moore.output
// CHECK: }
// CHECK: moore.class.classdecl @"testModule9::testModuleClass" {
// CHECK:   moore.class.propertydecl @a : !moore.i32
// CHECK:   moore.class.methoddecl @myReturn : (!moore.class<@"testModule9::testModuleClass">) -> !moore.i32
// CHECK: }
// CHECK: func.func private @"testModule9::testModuleClass::myReturn"([[this_ref:%.+]]: !moore.class<@"testModule9::testModuleClass">) -> !moore.i32 {
// CHECK:   [[prop_ref:%.+]] = moore.class.property_ref [[this_ref]][@a] : <@"testModule9::testModuleClass"> -> <i32>
// CHECK:   [[read_val:%.+]] = moore.read [[prop_ref]] : <i32>
// CHECK:   return [[read_val]] : !moore.i32
// CHECK: }
// CHECK: moore.class.classdecl @"testModule9::testModuleClass2" extends @"testModule9::testModuleClass" {
// CHECK:   moore.class.methoddecl @returnA : (!moore.class<@"testModule9::testModuleClass2">) -> !moore.i32
// CHECK: }
// CHECK: func.func private @"testModule9::testModuleClass2::returnA"([[this_ref2:%.+]]: !moore.class<@"testModule9::testModuleClass2">) -> !moore.i32 {
// CHECK:   [[upcast_ref:%.+]] = moore.class.upcast [[this_ref2]] : <@"testModule9::testModuleClass2"> to <@"testModule9::testModuleClass">
// CHECK:   [[call_myReturn:%.+]] = call @"testModule9::testModuleClass::myReturn"([[upcast_ref]]) : (!moore.class<@"testModule9::testModuleClass">) -> !moore.i32
// CHECK:   return [[call_myReturn]] : !moore.i32
// CHECK: }

module testModule9;

    class testModuleClass;
       int a;
       function int myReturn();
          return a;
       endfunction; // myReturn
    endclass // testModuleClass

   class testModuleClass2 extends testModuleClass;
       function int returnA();
          return myReturn();
       endfunction
   endclass // testModuleClass2

    testModuleClass2 t;
    int result;
    initial begin
        t = new;
        result = t.returnA();
    end

endmodule

/// Check forward declarations

// CHECK-LABEL:  moore.class.classdecl @testModuleClass {
// CHECK:    moore.class.methoddecl @testFunction : (!moore.class<@testModuleClass>, !moore.i32) -> !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testModuleClass::testFunction"(%arg0: !moore.class<@testModuleClass>, %arg1: !moore.i32) -> !moore.i32 {
// CHECK:    return %arg1 : !moore.i32
// CHECK:  }

class testModuleClass;
    extern function int testFunction(int a);
endclass

function int testModuleClass::testFunction(int a);
    return a;
endfunction
