// RUN: circt-opt %s -verify-diagnostics | circt-opt | FileCheck %s

// CHECK-LABEL: hw.module @parameters<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8) {
hw.module @parameters<p1: i42 = 17, p2: i1>(%arg0: i8) -> (out: i8) {
  hw.output %arg0 : i8
}

// CHECK-LABEL: hw.module @UseParameterized(
hw.module @UseParameterized(%a: i8) -> (xx: i8, yy: i8, zz: i8) {
  // CHECK: %inst1.out = hw.instance "inst1" @parameters<p1: i42 = 4, p2: i1 = false>(arg0:
  %r0 = hw.instance "inst1" @parameters<p1: i42 = 4, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  // CHECK: %inst2.out = hw.instance "inst2" @parameters<p1: i42 = 11, p2: i1 = true>(arg0:
  %r1 = hw.instance "inst2" @parameters<p1: i42 = 11, p2: i1 = 1>(arg0: %a: i8) -> (out: i8)

  // CHECK: %inst3.out = hw.instance "inst3" @parameters<p1: i42 = 17, p2: i1 = false>(arg0:
  %r2 = hw.instance "inst3" @parameters<p1: i42 = 17, p2: i1 = 0>(arg0: %a: i8) -> (out: i8)

  hw.output %r0, %r1, %r2: i8, i8, i8
}


// CHECK-LABEL: hw.module.extern @NoArg<param: i42>()
hw.module.extern @NoArg<param: i42>()

// CHECK-LABEL: hw.module @UseParameters<p1: i42>() {
hw.module @UseParameters<p1: i42>() {
  // CHECK: hw.instance "verbatimparam" @NoArg<param: i42 =
  // CHECK-SAME: #hw.param.verbatim<"\22FOO\22">>() -> () 
  hw.instance "verbatimparam" @NoArg<param: i42 = #hw.param.verbatim<"\"FOO\"">>() -> ()

  // CHECK: hw.instance "verbatimparam" @NoArg<param: i42 =
  // CHECK-SAME: #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> () 
  hw.instance "verbatimparam" @NoArg<param: i42 = #hw.param.expr.add<#hw.param.verbatim<"xxx">, 17>>() -> () 
  hw.output
}

// CHECK-LABEL: hw.module @addParam
hw.module @addParam<p1: i4, p2: i4>()
  -> (o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4, o: i4) {
  // CHECK-NEXT: %0 = hw.param.value i4 = 6
  %0 = hw.param.value i4 = #hw.param.expr.add<1, 2, 3>
  // CHECK-NEXT: %1 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p1">, 4>
  %1 = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, 4, #hw.param.decl.ref<"p1">>
  // CHECK-NEXT: %2 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, 6>
  %2 = hw.param.value i4 = #hw.param.expr.add<2, 4, #hw.param.decl.ref<"p1">>
  // CHECK-NEXT: %3 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p2">, 4>
  %3 = hw.param.value i4 = #hw.param.expr.add<#hw.param.decl.ref<"p1">, 4, #hw.param.decl.ref<"p2">>

  // CHECK-NEXT: %4 = hw.param.value i4 =
  // CHECK-SAME:     #hw.param.expr.add<#hw.param.decl.ref<"p1">, #hw.param.decl.ref<"p1">, 4>
  %4 = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.add<#hw.param.decl.ref<"p1">, 4>, #hw.param.decl.ref<"p1">>
 
  // CHECK-NEXT: %5 = hw.param.value i4 = #hw.param.decl.ref<"p1">
  %5 = hw.param.value i4 = #hw.param.expr.add<8, #hw.param.decl.ref<"p1">, 8>

  // CHECK-NEXT: %6 = hw.param.value i4 = 0
  %6 = hw.param.value i4 = #hw.param.expr.mul<8, #hw.param.decl.ref<"p1">, 8>

  // CHECK-NEXT: %7 = hw.param.value i4 = #hw.param.expr.mul<#hw.param.decl.ref<"p1">, 4>
  %7 = hw.param.value i4 = #hw.param.expr.shl<#hw.param.decl.ref<"p1">, 2>

  // CHECK-NEXT: %8 = hw.param.value i4 = #hw.param.expr.add<#hw.param.expr.mul<#hw.param.decl.ref<"p2">, #hw.param.decl.ref<"p1">, 2>, #hw.param.expr.mul<#hw.param.decl.ref<"p2">, 6>>
  %8 = hw.param.value i4 = #hw.param.expr.mul<#hw.param.expr.add<#hw.param.decl.ref<"p1">, 3>, 2, #hw.param.decl.ref<"p2">>

  hw.output %0, %1, %2, %3, %4, %5, %6, %7, %8
     : i4, i4, i4, i4, i4, i4, i4, i4, i4
}

