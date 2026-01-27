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
// CHECK-NEXT:   moore.class.propertydecl @localAutoI32 : !moore.i32
// CHECK: }
// CHECK-LABEL: moore.global_variable @"PropertyCombo::protStatL18" : !moore.l18

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
// CHECK: }
// CHECK: func.func private @"testModule9::testModuleClass::myReturn"([[this_ref:%.+]]: !moore.class<@"testModule9::testModuleClass">) -> !moore.i32 {
// CHECK:   [[prop_ref:%.+]] = moore.class.property_ref [[this_ref]][@a] : <@"testModule9::testModuleClass"> -> <i32>
// CHECK:   [[read_val:%.+]] = moore.read [[prop_ref]] : <i32>
// CHECK:   return [[read_val]] : !moore.i32
// CHECK: }
// CHECK: moore.class.classdecl @"testModule9::testModuleClass2" extends @"testModule9::testModuleClass" {
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

/// Check that calls to new by classes with ctor call the ctor.

// CHECK-LABEL:  moore.module @testModule10() {
// CHECK:    moore.procedure initial {
// CHECK:      [[NEW:%.+]] = moore.class.new : <@"testModule10::testModuleClass">
// CHECK:      [[CONST:%.+]] = moore.constant 3 : i32
// CHECK:      func.call @"testModule10::testModuleClass::new"([[NEW]], [[CONST]]) : (!moore.class<@"testModule10::testModuleClass">, !moore.i32) -> ()
// CHECK:      [[VAR:%.+]] = moore.variable [[NEW]] : <class<@"testModule10::testModuleClass">>
// CHECK:      moore.return
// CHECK:    }
// CHECK:    moore.output
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModule10::testModuleClass" {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testModule10::testModuleClass::new"(%arg0: !moore.class<@"testModule10::testModuleClass">, %arg1: !moore.i32) {
// CHECK:    [[NEW:%.+]] = moore.variable %arg1 : <i32>
// CHECK:    [[RNEW:%.+]] = moore.read [[NEW]] : <i32>
// CHECK:    moore.blocking_assign [[NEW]], [[RNEW]] : i32
// CHECK:    return
// CHECK:  }

module testModule10;

    class testModuleClass;
       int a;
        function new(int a);
           a = a;
        endfunction
    endclass // testModuleClass

    initial begin
       static testModuleClass t = new(3);
    end

endmodule

/// Check that calls to new by classes with super ctor call the ctor.

// CHECK-LABEL:  moore.class.classdecl @testModuleClass2 {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testModuleClass2::new"(%arg0: !moore.class<@testModuleClass2>, %arg1: !moore.i32) {
// CHECK:    [[A:%.+]] = moore.class.property_ref %arg0[@a] : <@testModuleClass2> -> <i32>
// CHECK:    moore.blocking_assign [[A]], %arg1 : i32
// CHECK:    return
// CHECK:  }
// CHECK:  moore.class.classdecl @testModuleClass3 extends @testModuleClass2 {
// CHECK:  }
// CHECK:  func.func private @"testModuleClass3::new"(%arg0: !moore.class<@testModuleClass3>, %arg1: !moore.i32) {
// CHECK:    [[UPCAST:%.+]] = moore.class.upcast %arg0 : <@testModuleClass3> to <@testModuleClass2>
// CHECK:    call @"testModuleClass2::new"([[UPCAST]], %arg1) : (!moore.class<@testModuleClass2>, !moore.i32) -> ()
// CHECK:    return
// CHECK:  }

class testModuleClass2;
    int a;
    function new(int a);
        this.a = a;
    endfunction
endclass // testModuleClass

class testModuleClass3 extends testModuleClass2;
    function new(int a);
        super.new(a);
    endfunction
endclass // testModuleClass

/// Check specialized class decl lowering

// CHECK-LABEL:  moore.module @testModuleParametrized() {
// CHECK:    [[T:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass">>
// CHECK:    [[T2:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass">>
// CHECK:    [[T3:%.+]] = moore.variable : <class<@"testModuleParametrized::testModuleClass_0">>
// CHECK:    moore.output
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass" {
// CHECK:    moore.class.propertydecl @a : !moore.l32
// CHECK:    moore.class.propertydecl @b : !moore.l4
// CHECK:  }
// CHECK:  moore.class.classdecl @"testModuleParametrized::testModuleClass_0" {
// CHECK:    moore.class.propertydecl @a : !moore.l16
// CHECK:    moore.class.propertydecl @b : !moore.l16
// CHECK:  }

module testModuleParametrized;

    class testModuleClass #(
        parameter int WIDTH=32,
        parameter int Other=16
    );
       logic [WIDTH-1:0] a;
       logic [Other-1:0] b;
    endclass // testModuleClass

   testModuleClass#(.WIDTH(32), .Other(4)) t;
   testModuleClass#(.WIDTH(32), .Other(4)) t2;
   testModuleClass#(.WIDTH(16)) t3;
endmodule

/// A test for getting a PR merged that drops elaboration-time constant AST nodes

// CHECK-LABEL:  moore.class.classdecl @testTypedClass extends @testClassType {
// CHECK-NEXT: moore.class.propertydecl @a : !moore.l1
// CHECK:  }

class testClassType #(
    parameter type t = logic
);
   typedef t bool;
endclass

class testTypedClass extends testClassType;
   bool a;
endclass

/// Check that LValues get generated for ClassProperty assignments

// CHECK-LABEL:  moore.class.classdecl @testLValueClass {
// CHECK:    moore.class.propertydecl @a : !moore.i32
// CHECK:  }
// CHECK:  func.func private @"testLValueClass::adder"(%arg0: !moore.class<@testLValueClass>) {
// CHECK:    [[LVAL:%.+]] = moore.class.property_ref %arg0[@a] : <@testLValueClass> -> <i32>
// CHECK:    [[RLVAL:%.+]] = moore.class.property_ref %arg0[@a] : <@testLValueClass> -> <i32>
// CHECK:    [[RVAL:%.+]] = moore.read [[RLVAL]] : <i32>
// CHECK:    [[CONST:%.+]] = moore.constant 1 : i32
// CHECK:    [[NEWVAL:%.+]] = moore.add [[RVAL]], [[CONST]] : i32
// CHECK:    moore.blocking_assign [[LVAL]], [[NEWVAL]] : i32
// CHECK:    return
// CHECK:  }

class testLValueClass;
int a;
function void adder;
    a = a + 1;
endfunction
endclass

/// Check that inheritance is enforced over specialized classes

// CHECK-LABEL:  moore.class.classdecl @GenericBar {
// CHECK:  }
// CHECK:  moore.class.classdecl @SpecializedFoo extends @GenericBar {
// CHECK:  }

class GenericBar #(int X=0, int Y=1, int Z=2); endclass
localparam x=3, y=4, z=5;

class SpecializedFoo extends GenericBar #(x,y,z); endclass

/// Check virtual attribute of methoddecl

// CHECK-LABEL: moore.class.classdecl @testClassVirtual {
// CHECK-NEXT:    moore.class.methoddecl @testFun -> @"testClassVirtual::testFun" : (!moore.class<@testClassVirtual>) -> ()
// CHECK:  }
// CHECK:  func.func private @"testClassVirtual::testFun"(%arg0: !moore.class<@testClassVirtual>) {
// CHECK:    return
// CHECK:  }

class testClassVirtual;
   virtual function void testFun();
   endfunction
endclass

/// Check virtual dispatch

// CHECK-LABEL: func.func private @testVirtualDispatch
// CHECK-SAME: (%arg0: !moore.class<@testClassVirtual>) {
// CHECK-NEXT:    [[VMETH:%.+]] = moore.vtable.load_method %arg0 : @testFun of <@testClassVirtual> -> (!moore.class<@testClassVirtual>) -> ()
// CHECK-NEXT:    call_indirect [[VMETH]](%arg0) : (!moore.class<@testClassVirtual>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

function void testVirtualDispatch (testClassVirtual t);
    t.testFun();
endfunction

/// Check pure virtual forward declarations

// CHECK-LABEL:  moore.class.classdecl @virtualFunctionClass {
// CHECK:    moore.class.methoddecl @subroutine : (!moore.class<@virtualFunctionClass>)
// CHECK:  }
// CHECK:  moore.class.classdecl @realFunctionClass implements [@virtualFunctionClass] {
// CHECK:    moore.class.methoddecl @subroutine -> @"realFunctionClass::subroutine" : (!moore.class<@realFunctionClass>)
// CHECK:  }
// CHECK:  func.func private @"realFunctionClass::subroutine"(%arg0: !moore.class<@realFunctionClass>) {
// CHECK:    return
// CHECK:  }

interface class virtualFunctionClass;
pure virtual function void subroutine;
endclass

class realFunctionClass implements virtualFunctionClass;
virtual function void subroutine; endfunction
endclass

/// Check that elaboration-time parameter accesses evaluate to constants

// CHECK-LABEL: moore.class.classdecl @parameterAccessClass

class parameterAccessClass #(int testParam = 1);
   extern function int testFunction();
endclass

// CHECK-LABEL: func.func private @"parameterAccessClass::testFunction
// CHECK-SAME: (%arg0: !moore.class<@parameterAccessClass>)
// CHECK: [[C0:%.*]] = moore.constant 7 : i32
// CHECK: [[C1:%.*]] = moore.constant 7 : i32
// CHECK: [[SUM:%.*]] = moore.add [[C0]], [[C1]] : i32
// CHECK: [[C3:%.*]] = moore.constant 7 : i32
// CHECK: [[SUM2:%.*]] = moore.add [[SUM]], [[C3]] : i32
// CHECK: return [[SUM2]] : !moore.i32

function int parameterAccessClass::testFunction();
   return this.testParam + testParam + parameterAccessClass::testParam;
endfunction

// CHECK-LABEL: func.func private @testFun() -> !moore.i32 {
// CHECK: [[VC:%.+]] = moore.variable : <class<@parameterAccessClass>>
// CHECK: [[READ:%.+]] = moore.read [[VC]] : <class<@parameterAccessClass>>
// CHECK: [[CALL:%.+]] = call @"parameterAccessClass::testFunction"([[READ]]) : (!moore.class<@parameterAccessClass>) -> !moore.i32
// CHECK: return [[CALL]] : !moore.i32

function int testFun;
   parameterAccessClass#(7) c;
   return c.testFunction();
endfunction

// Check method forward declarations seeing all class properties.

// CHECK-LABEL:  moore.class.classdecl @methodProtoClass {
// CHECK: moore.class.propertydecl @mytestvar : !moore.i32
// CHECK: }

class methodProtoClass;
   extern function int testFunction;
   int mytestvar;
endclass

// CHECK-LABEL: func.func private @"methodProtoClass::testFunction"(
// CHECK-SAME: %arg0: !moore.class<@methodProtoClass>) -> !moore.i32 {
// CHECK: [[PREF:%.+]] = moore.class.property_ref %arg0[@mytestvar] : <@methodProtoClass> -> <i32>
// CHECK: [[RREF:%.+]] = moore.read [[PREF]] : <i32>
// CHECK: return [[RREF]] : !moore.i32
// CHECK: }

function int methodProtoClass::testFunction();
   return mytestvar;
endfunction


// Check static member declarations are emitted as global variables
// Also check that name resolution properly prefixes them

// CHECK-LABEL: moore.global_variable @member : !moore.i32
static int member;

// CHECK-LABEL: moore.class.classdecl @staticMemberClass {
// CHECK-NEXT: }
class staticMemberClass;

// CHECK-LABEL:   moore.global_variable @"staticMemberClass::member" : !moore.i32
   static int member;

// CHECK-LABEL:  func.func private @next_member() -> !moore.i32 {
// CHECK:    [[MEMVAR:%.+]] = moore.get_global_variable @"staticMemberClass::member" : <i32>
// CHECK:    [[RVAR:%.+]] = moore.read [[MEMVAR]] : <i32>
// CHECK:    return [[RVAR]] : !moore.i32
// CHECK:  }

    static function int next_member();
        return member;
    endfunction
endclass
