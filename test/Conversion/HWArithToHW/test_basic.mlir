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

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1]], %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.add %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i2, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i2, i32
// CHECK:   %[[SIUI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %siui = hwarith.add %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i2, i32
// CHECK:   %[[UISI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %uisi = hwarith.add %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[UIUI_RES:.*]] = comb.add %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uiui = hwarith.add %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui33

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
// CHECK:   %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i34) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i34) -> i32
// CHECK:   %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui33) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @sub(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SIGN_BIT_OP0_1:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP0_1]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1_1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_BIT_OP1_1]], %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.sub %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0_2:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i2, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i2, i32
// CHECK:   %[[SIUI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %siui = hwarith.sub %op0Signed, %op1Unsigned : (si32, ui32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant 0 : i2
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i2, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i2
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i2, i32
// CHECK:   %[[UISI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i34
  %uisi = hwarith.sub %op0Unsigned, %op1Signed : (ui32, si32) -> si34

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[UIUI_RES:.*]] = comb.sub %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uiui = hwarith.sub %op0Unsigned, %op1Unsigned : (ui32, ui32) -> si33

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
// CHECK:   %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i34) -> i32
  %siuiOut = hwarith.cast %siui : (si34) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i34) -> i32
  %uisiOut = hwarith.cast %uisi : (si34) -> i32
// CHECK:   %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i33) -> i32
  %uiuiOut = hwarith.cast %uiui : (si33) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @mul(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {

  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:  %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i32, i32
// CHECK:  %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i32, i32
// CHECK:  %[[SISI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %sisi = hwarith.mul %op0Signed, %op1Signed : (si32, si32) -> si64

// CHECK:  %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP0]] : (i1) -> i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op0 : i32, i32
// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i32, i32
// CHECK:  %[[SIUI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %siui = hwarith.mul %op0Signed, %op1Unsigned : (si32, ui32) -> si64

// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i32, i32
// CHECK:  %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:  %[[SIGN_EXTEND:.*]] = comb.replicate %[[SIGN_BIT_OP1]] : (i1) -> i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[SIGN_EXTEND]], %op1 : i32, i32
// CHECK:  %[[UISI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %uisi = hwarith.mul %op0Unsigned, %op1Signed : (ui32, si32) -> si64

// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i32, i32
// CHECK:  %[[ZERO_EXTEND:.*]] = hw.constant 0 : i32
// CHECK:  %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i32, i32
// CHECK:  %[[UIUI_RES:.*]] = comb.mul %[[OP0_PADDED]], %[[OP1_PADDED]] : i64
  %uiui = hwarith.mul %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui64

// CHECK:  %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i64) -> i32
  %sisiOut = hwarith.cast %sisi : (si64) -> i32
// CHECK:  %[[SIUI_OUT:.*]] = comb.extract %[[SIUI_RES]] from 0 : (i64) -> i32
  %siuiOut = hwarith.cast %siui : (si64) -> i32
// CHECK:  %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i64) -> i32
  %uisiOut = hwarith.cast %uisi : (si64) -> i32
// CHECK:  %[[UIUI_OUT:.*]] = comb.extract %[[UIUI_RES]] from 0 : (i64) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui64) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_OUT]], %[[UISI_OUT]], %[[UIUI_OUT]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}

// -----

// CHECK: hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
hw.module @div(%op0: i32, %op1: i32) -> (sisi: i32, siui: i32, uisi: i32, uiui: i32) {
  %op0Signed = hwarith.cast %op0 : (i32) -> si32
  %op0Unsigned = hwarith.cast %op0 : (i32) -> ui32
  %op1Signed = hwarith.cast %op1 : (i32) -> si32
  %op1Unsigned = hwarith.cast %op1 : (i32) -> ui32

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %0, %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %2, %op1 : i1, i32
// CHECK:   %[[SISI_RES:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %sisi = hwarith.div %op0Signed, %op1Signed : (si32, si32) -> si33

// CHECK:   %[[SIGN_BIT_OP0:.*]] = comb.extract %op0 from 31 : (i32) -> i1
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %5, %op0 : i1, i32
// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op1 : i1, i32
// CHECK:   %[[SIUI_RES_IMM:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
// CHECK:   %[[SIUI_RES:.*]] = comb.extract %[[SIUI_RES_IMM]] from 0 : (i33) -> i32
  %siui = hwarith.div %op0Signed, %op1Unsigned : (si32, ui32) -> si32

// CHECK:   %[[ZERO_EXTEND:.*]] = hw.constant false
// CHECK:   %[[OP0_PADDED:.*]] = comb.concat %[[ZERO_EXTEND]], %op0 : i1, i32
// CHECK:   %[[SIGN_BIT_OP1:.*]] = comb.extract %op1 from 31 : (i32) -> i1
// CHECK:   %[[OP1_PADDED:.*]] = comb.concat %11, %op1 : i1, i32
// CHECK:   %[[UISI_RES:.*]] = comb.divs %[[OP0_PADDED]], %[[OP1_PADDED]] : i33
  %uisi = hwarith.div %op0Unsigned, %op1Signed : (ui32, si32) -> si33

// CHECK:   %[[UIUI_RES:.*]] = comb.divu %op0, %op1 : i32
  %uiui = hwarith.div %op0Unsigned, %op1Unsigned : (ui32, ui32) -> ui32

// CHECK:   %[[SISI_OUT:.*]] = comb.extract %[[SISI_RES]] from 0 : (i33) -> i32
  %sisiOut = hwarith.cast %sisi : (si33) -> i32
  %siuiOut = hwarith.cast %siui : (si32) -> i32
// CHECK:   %[[UISI_OUT:.*]] = comb.extract %[[UISI_RES]] from 0 : (i33) -> i32
  %uisiOut = hwarith.cast %uisi : (si33) -> i32
  %uiuiOut = hwarith.cast %uiui : (ui32) -> i32

// CHECK:   hw.output %[[SISI_OUT]], %[[SIUI_RES]], %[[UISI_OUT]], %[[UIUI_RES]] : i32, i32, i32, i32
  hw.output %sisiOut, %siuiOut, %uisiOut, %uiuiOut : i32, i32, i32, i32
}
