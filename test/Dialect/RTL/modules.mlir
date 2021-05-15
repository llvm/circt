// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

module {
  rtl.module @B(%a: i1) -> (%nameOfPortInSV: i1, i1) {
    %0 = comb.or %a, %a : i1
    %1 = comb.and %a, %a : i1
    rtl.output %0, %1: i1, i1
  }

  // CHECK-LABEL: rtl.module @B(%a: i1) -> (%nameOfPortInSV: i1, i1)
  // CHECK-NEXT:    %0 = comb.or %a, %a : i1
  // CHECK-NEXT:    %1 = comb.and %a, %a : i1
  // CHECK-NEXT:    rtl.output %0, %1 : i1, i1

  rtl.module.extern @C(%a: i1) -> (i1, i1) attributes {argNames=["nameOfPortInSV"]}
  // CHECK-LABEL: rtl.module.extern @C(%nameOfPortInSV: i1) -> (i1, i1)
  // CHECK-NOT: {

  rtl.module.extern @explicitResultName() -> (%x: i1) attributes {resultNames=["FOO"]}
  // CHECK-LABEL: rtl.module.extern @explicitResultName() -> (%FOO: i1)

  rtl.module.extern @D_ATTR(%a: i1) -> (i1, i1) attributes {filename = "test.v", parameters = {DEFAULT = 0 : i64}}

  // CHECK-LABEL: rtl.module.extern @D_ATTR(%a: i1) -> (i1, i1) attributes {filename = "test.v", parameters = {DEFAULT = 0 : i64}}
  // CHECK-NOT: {

  rtl.module @A(%d: i1, %e: !rtl.inout<i1>) -> (i1, i1) {
    // Instantiate @B as a HW module with result-as-output sementics
    %r1, %r2 = rtl.instance "b1" @B(%d) : (i1) -> (i1, i1)
    // Instantiate @C with a public symbol on the instance
    %f, %g = rtl.instance "c1" sym @E @C(%d) : (i1) -> (i1, i1)
    // Connect the inout port with %f
    sv.connect %e, %f : i1
    // Output values
    rtl.output %g, %r1 : i1, i1
  }
  // CHECK-LABEL: rtl.module @A(%d: i1, %e: !rtl.inout<i1>) -> (i1, i1)
  // CHECK-NEXT:  %b1.nameOfPortInSV, %b1.1 = rtl.instance "b1" @B(%d) : (i1) -> (i1, i1)
  // CHECK-NEXT:  %c1.0, %c1.1 = rtl.instance "c1" sym @E @C(%d) : (i1) -> (i1, i1)

  rtl.module @AnyType1(%a: vector< 3 x i8 >) { }
  // CHECK-LABEL: rtl.module @AnyType1(%a: vector<3xi8>)
  
  // CHECK-LABEL: rtl.module @AnyTypeInstance()
  rtl.module @AnyTypeInstance() {
    %vec = constant dense <0> : vector<3xi8>
    rtl.instance "anyType1" @AnyType1(%vec) : (vector<3xi8>) -> ()
  }

  // CHECK:       %cst = constant dense<0> : vector<3xi8>
  // CHECK-NEXT:  rtl.instance "anyType1" @AnyType1(%cst) : (vector<3xi8>) -> ()

  rtl.generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
  rtl.module.generated @genmod1, @MEMORY() -> (%FOOBAR: i1) attributes {write_latency=1, read_latency=1, ports=["read","write"]}
  // CHECK-LABEL: rtl.generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
  // CHECK-NEXT: rtl.module.generated @genmod1, @MEMORY() -> (%FOOBAR: i1) attributes {ports = ["read", "write"], read_latency = 1 : i64, write_latency = 1 : i64}


  // CHECK-LABEL: rtl.module.extern @AnonArg(i42)
  rtl.module.extern @AnonArg(i42)
}
