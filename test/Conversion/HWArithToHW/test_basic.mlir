// RUN: circt-opt -split-input-file -lower-hwarith-to-hw %s | FileCheck %s

// CHECK: hw.module @constant() -> (out: i32) {
// CHECK:   %c0_i32 = hw.constant 0 : i32
// CHECK:   hw.output %c0_i32 : i32

hw.module @constant() -> (out: i32) {
  %0 = hwarith.constant 0 : si32
  %out = hwarith.cast %0 : (si32) -> i32
  hw.output %out : i32
}

// -----

// CHECK: hw.module @add(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @add(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %0 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %1 = comb.concat %0, %op0 : i1, i32
// CHECK:   %2 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %3 = comb.concat %2, %op1 : i1, i32
// CHECK:   %4 = comb.add %1, %3 : i33
  %sisi = hwarith.add %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %5 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %6 = comb.replicate %5 : (i1) -> i2
// CHECK:   %7 = comb.concat %6, %op0 : i2, i32
// CHECK:   %c0_i2 = hw.constant 0 : i2
// CHECK:   %8 = comb.concat %c0_i2, %op1 : i2, i32
// CHECK:   %9 = comb.add %7, %8 : i34
  %siui = hwarith.add %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %c0_i2_0 = hw.constant 0 : i2
// CHECK:   %10 = comb.concat %c0_i2_0, %op0 : i2, i32
// CHECK:   %11 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %12 = comb.replicate %11 : (i1) -> i2
// CHECK:   %13 = comb.concat %12, %op1 : i2, i32
// CHECK:   %14 = comb.add %10, %13 : i34
  %uisi = hwarith.add %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %false = hw.constant false
// CHECK:   %15 = comb.concat %false, %op0 : i1, i32
// CHECK:   %false_1 = hw.constant false
// CHECK:   %16 = comb.concat %false_1, %op1 : i1, i32
// CHECK:   %17 = comb.add %15, %16 : i33
  %uiui = hwarith.add %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui33

// CHECK:   %18 = comb.extract %4 from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
// CHECK:   %19 = comb.extract %9 from 0 : (i34) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
// CHECK:   %20 = comb.extract %14 from 0 : (i34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
// CHECK:   %21 = comb.extract %17 from 0 : (i33) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui33) -> i32

// CHECK:   hw.output %18, %19, %20, %21 : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %0 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %1 = comb.concat %0, %op0 : i1, i32
// CHECK:   %2 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %3 = comb.concat %2, %op1 : i1, i32
// CHECK:   %4 = comb.sub %1, %3 : i33
  %sisi = hwarith.sub %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %5 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %6 = comb.replicate %5 : (i1) -> i2
// CHECK:   %7 = comb.concat %6, %op0 : i2, i32
// CHECK:   %c0_i2 = hw.constant 0 : i2
// CHECK:   %8 = comb.concat %c0_i2, %op1 : i2, i32
// CHECK:   %9 = comb.sub %7, %8 : i34
  %siui = hwarith.sub %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %c0_i2_0 = hw.constant 0 : i2
// CHECK:   %10 = comb.concat %c0_i2_0, %op0 : i2, i32
// CHECK:   %11 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %12 = comb.replicate %11 : (i1) -> i2
// CHECK:   %13 = comb.concat %12, %op1 : i2, i32
// CHECK:   %14 = comb.sub %10, %13 : i34
  %uisi = hwarith.sub %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %false = hw.constant false
// CHECK:   %15 = comb.concat %false, %op0 : i1, i32
// CHECK:   %false_1 = hw.constant false
// CHECK:   %16 = comb.concat %false_1, %op1 : i1, i32
// CHECK:   %17 = comb.sub %15, %16 : i33
  %uiui = hwarith.sub %op0Unsigned, %op1Unsigned : (ui32, ui32) -> si33

// CHECK:   %18 = comb.extract %4 from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
// CHECK:   %19 = comb.extract %9 from 0 : (i34) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
// CHECK:   %20 = comb.extract %14 from 0 : (i34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
// CHECK:   %21 = comb.extract %17 from 0 : (i33) -> i32
  %uiuiOut = hwarith.cast %uiui : (si33) -> i32

// CHECK:   hw.output %18, %19, %20, %21 : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {

  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:  %0 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %1 = comb.replicate %0 : (i1) -> i32
// CHECK:  %2 = comb.concat %1, %op0 : i32, i32
// CHECK:  %3 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %4 = comb.replicate %3 : (i1) -> i32
// CHECK:  %5 = comb.concat %4, %op1 : i32, i32
// CHECK:  %6 = comb.mul %2, %5 : i64
  %sisi = hwarith.mul %op0Signed, %op1Signed : (si32, si32) -> si64

// CHECK:  %7 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %8 = comb.replicate %7 : (i1) -> i32
// CHECK:  %9 = comb.concat %8, %op0 : i32, i32
// CHECK:  %c0_i32 = hw.constant 0 : i32
// CHECK:  %10 = comb.concat %c0_i32, %op1 : i32, i32
// CHECK:  %11 = comb.mul %9, %10 : i64
  %siui = hwarith.mul %op0Signed, %op1Unsigned : (si32, ui32) -> si64

// CHECK:  %c0_i32_0 = hw.constant 0 : i32
// CHECK:  %12 = comb.concat %c0_i32_0, %op0 : i32, i32
// CHECK:  %13 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %14 = comb.replicate %13 : (i1) -> i32
// CHECK:  %15 = comb.concat %14, %op1 : i32, i32
// CHECK:  %16 = comb.mul %12, %15 : i64
  %uisi = hwarith.mul %op0Unsigned, %op1Signed : (ui32, si32) -> si64

// CHECK:  %c0_i32_1 = hw.constant 0 : i32
// CHECK:  %17 = comb.concat %c0_i32_1, %op0 : i32, i32
// CHECK:  %c0_i32_2 = hw.constant 0 : i32
// CHECK:  %18 = comb.concat %c0_i32_2, %op1 : i32, i32
// CHECK:  %19 = comb.mul %17, %18 : i64
  %uiui = hwarith.mul %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui64

// CHECK:  %20 = comb.extract %6 from 0 : (i64) -> i32
  %sisiOut = hwarith.cast %sisi : (si64) -> i32
// CHECK:  %21 = comb.extract %11 from 0 : (i64) -> i32
  %siuiOut = hwarith.cast %siui : (si64) -> i32
// CHECK:  %22 = comb.extract %16 from 0 : (i64) -> i32
  %uisiOut = hwarith.cast %uisi : (si64) -> i32
// CHECK:  %23 = comb.extract %19 from 0 : (i64) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui64) -> i32

// CHECK:  hw.output %20, %21, %22, %23 : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %0 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %1 = comb.concat %0, %op0 : i1, i32
// CHECK:   %2 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %3 = comb.concat %2, %op1 : i1, i32
// CHECK:   %4 = comb.divs %1, %3 : i33
  %sisi = hwarith.div %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %5 = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %6 = comb.concat %5, %op0 : i1, i32
// CHECK:   %false = hw.constant false
// CHECK:   %7 = comb.concat %false, %op1 : i1, i32
// CHECK:   %8 = comb.divs %6, %7 : i33
// CHECK:   %9 = comb.extract %8 from 0 : (i33) -> i32
  %siui = hwarith.div %op0Signed, %op1Unsigned : (si32, ui32) -> si32

// CHECK:   %false_0 = hw.constant false
// CHECK:   %10 = comb.concat %false_0, %op0 : i1, i32
// CHECK:   %11 = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %12 = comb.concat %11, %op1 : i1, i32
// CHECK:   %13 = comb.divs %10, %12 : i33
  %uisi = hwarith.div %op0Unsigned, %op1Signed : (ui32, si32) -> si33

// CHECK:   %14 = comb.divu %op0, %op1 : i32
  %uiui = hwarith.div %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui32

// CHECK:   %15 = comb.extract %4 from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
  %siuiOut = hwarith.cast %siui : (si32) -> i32
// CHECK:   %16 = comb.extract %13 from 0 : (i33) -> i32
  %uisiOut = hwarith.cast %uisi : (si33) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui32) -> i32

// CHECK:   hw.output %15, %9, %16, %14 : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}
