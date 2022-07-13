// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl.module(firrtl-lower-bitindex))'  %s | FileCheck %s

firrtl.circuit "Test"  {

// CHECK-LABEL: firrtl.module @Test1
firrtl.module @Test1(in %x: !firrtl.uint<4>, in %y: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {

  // CHECK: [[WRAPPER:%.+]] = firrtl.wire   : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %1 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %2 = firrtl.cat %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %3 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %4 = firrtl.cat %3, %2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %5 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %6 = firrtl.cat %5, %4 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %out, %6 : !firrtl.uint<4>
  // CHECK: %7 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %7, %y : !firrtl.uint<1>


  %0 = firrtl.bitindex %out[0] : !firrtl.uint<4>
  firrtl.strictconnect %0, %y : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Test2
firrtl.module @Test2(in %clock: !firrtl.clock, in %x: !firrtl.uint<4>, in %y: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {

  // CHECK: %r = firrtl.reg interesting_name %clock  : !firrtl.uint<4>
  // CHECK: %r_0 = firrtl.wire   {name = "r"} : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.bits %r 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %1 = firrtl.subindex %r_0[0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %1, %0 : !firrtl.uint<1>
  // CHECK: %2 = firrtl.bits %r 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %3 = firrtl.subindex %r_0[1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %3, %2 : !firrtl.uint<1>
  // CHECK: %4 = firrtl.bits %r 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %5 = firrtl.subindex %r_0[2] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %5, %4 : !firrtl.uint<1>
  // CHECK: %6 = firrtl.bits %r 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %7 = firrtl.subindex %r_0[3] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %7, %6 : !firrtl.uint<1>
  // CHECK: %8 = firrtl.subindex %r_0[0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %9 = firrtl.subindex %r_0[1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %10 = firrtl.cat %9, %8 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %11 = firrtl.subindex %r_0[2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %12 = firrtl.cat %11, %10 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %13 = firrtl.subindex %r_0[3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %14 = firrtl.cat %13, %12 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %r, %14 : !firrtl.uint<4>
  // CHECK: %15 = firrtl.subindex %r_0[0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.when %en {
  // CHECK:   firrtl.strictconnect %15, %y : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.strictconnect %out, %r : !firrtl.uint<4>


  %r = firrtl.reg interesting_name %clock  : !firrtl.uint<4>
  %0 = firrtl.bitindex %r[0] : !firrtl.uint<4>
  firrtl.when %en {
    firrtl.strictconnect %0, %y : !firrtl.uint<1>
  }
  firrtl.strictconnect %out, %r : !firrtl.uint<4>
}





// CHECK-LABEL: firrtl.module @Test
firrtl.module @Test(in %x: !firrtl.uint<4>, in %y: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {
  // CHECK: [[WRAPPER:%.+]] = firrtl.wire   : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %1 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %2 = firrtl.cat %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %3 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %4 = firrtl.cat %3, %2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %5 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %6 = firrtl.cat %5, %4 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %out, %6 : !firrtl.uint<4>
  // CHECK: %7 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %8 = firrtl.bits %x 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %9 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %9, %8 : !firrtl.uint<1>
  // CHECK: %10 = firrtl.bits %x 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %11 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %11, %10 : !firrtl.uint<1>
  // CHECK: %12 = firrtl.bits %x 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %13 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %13, %12 : !firrtl.uint<1>
  // CHECK: %14 = firrtl.bits %x 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %15 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %15, %14 : !firrtl.uint<1>
  // CHECK: firrtl.strictconnect %7, %y : !firrtl.uint<1>


  %0 = firrtl.bitindex %out[2] : !firrtl.uint<4>
  firrtl.strictconnect %out, %x : !firrtl.uint<4>
  firrtl.strictconnect %0, %y : !firrtl.uint<1>
}


// CHECK-LABEL: firrtl.module @Test4
firrtl.module @Test4(in %x: !firrtl.uint<4>, in %y: !firrtl.uint<1>, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {

  // CHECK: [[WRAPPER:%.+]] = firrtl.wire   : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %1 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %2 = firrtl.cat %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %3 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %4 = firrtl.cat %3, %2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %5 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %6 = firrtl.cat %5, %4 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %out, %6 : !firrtl.uint<4>
  // CHECK: %7 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %8 = firrtl.bits %x 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %9 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %9, %8 : !firrtl.uint<1>
  // CHECK: %10 = firrtl.bits %x 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %11 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %11, %10 : !firrtl.uint<1>
  // CHECK: %12 = firrtl.bits %x 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %13 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %13, %12 : !firrtl.uint<1>
  // CHECK: %14 = firrtl.bits %x 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %15 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %15, %14 : !firrtl.uint<1>
  // CHECK: %16 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %7, %16 : !firrtl.uint<1>


  %0 = firrtl.bitindex %out[0] : !firrtl.uint<4>
  firrtl.strictconnect %out, %x : !firrtl.uint<4>
  %1 = firrtl.bits %out 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  firrtl.strictconnect %0, %1 : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Test5
firrtl.module @Test5(in %x: !firrtl.uint<4>, in %y: !firrtl.uint<1>, in %en: !firrtl.uint<1>, in %en_2: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {

  // CHECK: [[WRAPPER:%.+]] = firrtl.wire   : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %1 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %2 = firrtl.cat %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %3 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %4 = firrtl.cat %3, %2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %5 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %6 = firrtl.cat %5, %4 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %out, %6 : !firrtl.uint<4>
  // CHECK: %7 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %8 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %9 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %10 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %11 = firrtl.bits %x 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %12 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %12, %11 : !firrtl.uint<1>
  // CHECK: %13 = firrtl.bits %x 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %14 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %14, %13 : !firrtl.uint<1>
  // CHECK: %15 = firrtl.bits %x 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %16 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %16, %15 : !firrtl.uint<1>
  // CHECK: %17 = firrtl.bits %x 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %18 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %18, %17 : !firrtl.uint<1>
  // CHECK: firrtl.when %en {
  // CHECK:   firrtl.strictconnect %10, %y : !firrtl.uint<1>
  // CHECK:   firrtl.when %en_2 {
  // CHECK:     firrtl.strictconnect %9, %y : !firrtl.uint<1>
  // CHECK:     firrtl.strictconnect %8, %y : !firrtl.uint<1>
  // CHECK:   }
  // CHECK: } else {
  // CHECK:   firrtl.strictconnect %9, %y : !firrtl.uint<1>
  // CHECK: }
  // CHECK: firrtl.strictconnect %7, %y : !firrtl.uint<1>

  %0 = firrtl.bitindex %out[3] : !firrtl.uint<4>
  %1 = firrtl.bitindex %out[2] : !firrtl.uint<4>
  %2 = firrtl.bitindex %out[1] : !firrtl.uint<4>
  %3 = firrtl.bitindex %out[0] : !firrtl.uint<4>
  firrtl.strictconnect %out, %x : !firrtl.uint<4>
  firrtl.when %en {
    firrtl.strictconnect %3, %y : !firrtl.uint<1>
    firrtl.when %en_2 {
      firrtl.strictconnect %2, %y : !firrtl.uint<1>
      firrtl.strictconnect %1, %y : !firrtl.uint<1>
    }
  } else {
    firrtl.strictconnect %2, %y : !firrtl.uint<1>
  }
  firrtl.strictconnect %0, %y : !firrtl.uint<1>
}

// CHECK-LABEL: firrtl.module @Test6
firrtl.module @Test6(in %x: !firrtl.uint<4>, in %y: !firrtl.uint<4>, out %out: !firrtl.uint<4>) {

  // CHECK: [[WRAPPER:%.+]] = firrtl.wire   : !firrtl.vector<uint<1>, 4>
  // CHECK: %0 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %1 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %2 = firrtl.cat %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  // CHECK: %3 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: %4 = firrtl.cat %3, %2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<3>
  // CHECK: %5 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: %6 = firrtl.cat %5, %4 : (!firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<4>
  // CHECK: firrtl.strictconnect %out, %6 : !firrtl.uint<4>
  // CHECK: %7 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: %8 = firrtl.bits %x 0 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %9 = firrtl.subindex [[WRAPPER]][0] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %9, %8 : !firrtl.uint<1>
  // CHECK: %10 = firrtl.bits %x 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %11 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %11, %10 : !firrtl.uint<1>
  // CHECK: %12 = firrtl.bits %x 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %13 = firrtl.subindex [[WRAPPER]][2] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %13, %12 : !firrtl.uint<1>
  // CHECK: %14 = firrtl.bits %x 3 to 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %15 = firrtl.subindex [[WRAPPER]][3] : !firrtl.vector<uint<1>, 4>
  // CHECK: firrtl.strictconnect %15, %14 : !firrtl.uint<1>
  // CHECK: %16 = firrtl.subindex [[WRAPPER]][1] : !firrtl.vector<uint<1>, 4>
  // CHECK: %17 = firrtl.bits %y 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK: %18 = firrtl.or %16, %17 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: firrtl.strictconnect %7, %18 : !firrtl.uint<1>


  %0 = firrtl.bitindex %out[0] : !firrtl.uint<4>
  firrtl.strictconnect %out, %x : !firrtl.uint<4>
  %1 = firrtl.bits %out 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  %2 = firrtl.bits %y 1 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  %3 = firrtl.or %1, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  firrtl.strictconnect %0, %3 : !firrtl.uint<1>
}



}

