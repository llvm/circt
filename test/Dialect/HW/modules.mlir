// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

hw.module @B(in %a: i1, out nameOfPortInSV: i1, out "": i1) {
  %0 = comb.or %a, %a : i1
  %1 = comb.and %a, %a : i1
  hw.output %0, %1: i1, i1
}

// CHECK-LABEL: hw.module @B(in %a : i1, out nameOfPortInSV : i1, out "" : i1)
// CHECK-NEXT:    %0 = comb.or %a, %a : i1
// CHECK-NEXT:    %1 = comb.and %a, %a : i1
// CHECK-NEXT:    hw.output %0, %1 : i1, i1

hw.module.extern @C(in %nameOfPortInSV: i1, out "": i1, out "": i1)
// CHECK-LABEL: hw.module.extern @C(in %nameOfPortInSV : i1, out "" : i1, out "" : i1)
// CHECK-NOT: {

hw.module.extern @explicitResultName(out FOO: i1)
// CHECK-LABEL: hw.module.extern @explicitResultName(out FOO : i1)

hw.module.extern @D_ATTR(in %a: i1, out "": i1, out "": i1) attributes {filename = "test.v", p = {DEFAULT = 0 : i64}}

// CHECK-LABEL: hw.module.extern @D_ATTR(in %a : i1, out "" : i1, out "" : i1) attributes {filename = "test.v", p = {DEFAULT = 0 : i64}}
// CHECK-NOT: {

hw.module @A(in %d: i1, out "": i1, out "": i1) {
  // Instantiate @B as a HW module with result-as-output sementics
  %r1, %r2 = hw.instance "b1" @B(a: %d: i1) -> (nameOfPortInSV: i1, "": i1)
  // Instantiate @C with a public symbol on the instance
  %f, %g = hw.instance "c1" sym @E @C(nameOfPortInSV: %d: i1) -> ("": i1, "": i1)
  // Output values
  hw.output %g, %r1 : i1, i1
}
// CHECK-LABEL: hw.module @A(in %d : i1, out "" : i1, out "" : i1)
// CHECK-NEXT:  %b1.nameOfPortInSV, %b1.1 = hw.instance "b1" @B(a: %d: i1) -> (nameOfPortInSV: i1, "": i1)
// CHECK-NEXT:  %c1.0, %c1.1 = hw.instance "c1" sym @E @C(nameOfPortInSV: %d: i1) -> ("": i1, "": i1)

hw.module @AnyType1(in %a: vector< 3 x i8 >) { }
// CHECK-LABEL: hw.module @AnyType1(in %a : vector<3xi8>)

// CHECK-LABEL: hw.module @AnyTypeInstance()
hw.module @AnyTypeInstance() {
  %vec = arith.constant dense <0> : vector<3xi8>
  hw.instance "anyType1" @AnyType1(a: %vec: vector<3xi8>) -> ()
}

// CHECK:       %cst = arith.constant dense<0> : vector<3xi8>
// CHECK-NEXT:  hw.instance "anyType1" @AnyType1(a: %cst: vector<3xi8>) -> ()

hw.generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
hw.module.generated @genmod1, @MEMORY(out FOOBAR: i1) attributes {write_latency=1, read_latency=1, ports=["read","write"]}
// CHECK-LABEL: hw.generator.schema @MEMORY, "Simple-Memory", ["ports", "write_latency", "read_latency"]
// CHECK-NEXT: hw.module.generated @genmod1, @MEMORY(out FOOBAR : i1) attributes {ports = ["read", "write"], read_latency = 1 : i64, write_latency = 1 : i64}

hw.module @SpecPlain() attributes {hw.specialization = #hw.specialization<"Orig">} {}
// CHECK-LABEL: hw.module @SpecPlain() attributes {hw.specialization = #hw.specialization<"Orig">}

hw.module @SpecEmptyParams() attributes {hw.specialization = #hw.specialization<"Orig", []>} {}
// CHECK-LABEL: hw.module @SpecEmptyParams() attributes {hw.specialization = #hw.specialization<"Orig">}

hw.module @SpecAltered() attributes {hw.specialization = #hw.specialization<"Orig", boundaryAltered>} {}
// CHECK-LABEL: hw.module @SpecAltered() attributes {hw.specialization = #hw.specialization<"Orig", boundaryAltered>}

hw.module @SpecEmptyParamsAltered() attributes {hw.specialization = #hw.specialization<"Orig", [], boundaryAltered>} {}
// CHECK-LABEL: hw.module @SpecEmptyParamsAltered() attributes {hw.specialization = #hw.specialization<"Orig", boundaryAltered>}

hw.module @SpecParams() attributes {hw.specialization = #hw.specialization<"Orig", [#hw.param.decl<"W": si32 = 16> : si32, #hw.param.decl<"U": ui8 = 255> : ui8, #hw.param.decl<"R": f64 = 1.5> : f64, #hw.param.decl<"S": none = "hi">, #hw.param.decl<"V": none = #hw.param.verbatim<"4'b10x1">>, #hw.param.decl<"O": none>], boundaryAltered>} {}
// CHECK-LABEL: hw.module @SpecParams() attributes {hw.specialization = #hw.specialization<"Orig", [#hw.param.decl<"W": si32 = 16> : si32, #hw.param.decl<"U": ui8 = 255> : ui8, #hw.param.decl<"R": f64 = 1.500000e+00> : f64, #hw.param.decl<"S": none = "hi">, #hw.param.decl<"V": none = #hw.param.verbatim<"4'b10x1">>, #hw.param.decl<"O": none>], boundaryAltered>}
