// RUN: circt-opt -split-input-file -hw-legalize-modules -verify-diagnostics %s | FileCheck %s

module attributes {circt.loweringOptions = "disallowPackedArrays"} {
hw.module @reject_arrays(in %arg0: i8, in %arg1: i8, in %arg2: i8,
                         in %arg3: i8, in %sel: i2, in %clock: i1,
                         out a: !hw.array<4xi8>) {
  %reg = sv.reg  : !hw.inout<array<4xi8>>
  sv.alwaysff(posedge %clock)  {
    %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8
    sv.passign %reg, %0 : !hw.array<4xi8>
  }

  // This needs full-on "legalize types" for the HW dialect.
  // expected-error @+1 {{unsupported packed array expression}}
  %1 = sv.read_inout %reg : !hw.inout<array<4xi8>>
  hw.output %1 : !hw.array<4xi8>
}
}

// -----
module attributes {circt.loweringOptions = "disallowPackedArrays"} {
// CHECK-LABEL: hw.module @array_create_get_comb
hw.module @array_create_get_comb(in %arg0: i8, in %arg1: i8, in %arg2: i8, in %arg3: i8,
                                 in %sel: i2, out a: i8) {
  // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
  // CHECK: sv.alwayscomb  {
  // CHECK:   sv.case casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.bpassign %casez_tmp, %arg0 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %arg1 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %arg2 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %arg3 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.array_create %arg3, %arg2, %arg1, %arg0 : i8

  // CHECK: %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>, i2

  // CHECK: hw.output %0 : i8
  hw.output %1 : i8
}

// CHECK-LABEL: hw.module @array_create_get_default
hw.module @array_create_get_default(in %arg0: i8, in %arg1: i8, in %arg2: i8, in %arg3: i8,
                            in %sel: i2) {
  // CHECK: sv.initial  {
  sv.initial {
    // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
    // CHECK:   %x_i8 = sv.constantX : i8
    // CHECK:   sv.case casez %sel : i2
    // CHECK:   case b00: {
    // CHECK:     sv.bpassign %casez_tmp, %arg0 : i8
    // CHECK:   }
    // CHECK:   case b01: {
    // CHECK:     sv.bpassign %casez_tmp, %arg1 : i8
    // CHECK:   }
    // CHECK:   case b10: {
    // CHECK:     sv.bpassign %casez_tmp, %arg2 : i8
    // CHECK:   }
    // CHECK:   default: {
    // CHECK:     sv.bpassign %casez_tmp, %x_i8 : i8
    // CHECK:   }
    %three_array = hw.array_create %arg2, %arg1, %arg0 : i8

    // CHECK:   %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
    %2 = hw.array_get %three_array[%sel] : !hw.array<3xi8>, i2

    // CHECK:   %1 = comb.icmp eq %0, %arg2 : i8
    // CHECK:   sv.if %1  {
    %cond = comb.icmp eq %2, %arg2 : i8
    sv.if %cond {
      sv.fatal 1
    }
  }
}

// CHECK-LABEL: hw.module @array_muxed_create_get_default
hw.module @array_muxed_create_get_default(in %arg0: i8, in %arg1: i8, in %arg2: i8, in %arg3: i8, in %arg4: i8, in %arg5: i8,
                            in %array_sel: i1, in %index_sel: i2) {
  // CHECK: sv.initial  {
  sv.initial {
    %three_array1 = hw.array_create %arg2, %arg1, %arg0 : i8
    %three_array2 = hw.array_create %arg5, %arg4, %arg3 : i8

    // CHECK: %0 = comb.mux %array_sel, %arg0, %arg3 : i8
    // CHECK: %1 = comb.mux %array_sel, %arg1, %arg4 : i8
    // CHECK: %2 = comb.mux %array_sel, %arg2, %arg5 : i8
    %muxed = comb.mux %array_sel, %three_array1, %three_array2 : !hw.array<3xi8>

    // CHECK: %x_i8 = sv.constantX : i8
    // CHECK: sv.case casez %index_sel : i2
    // CHECK: case b00: {
    // CHECK:   sv.bpassign %casez_tmp, %0 : i8
    // CHECK: }
    // CHECK: case b01: {
    // CHECK:   sv.bpassign %casez_tmp, %1 : i8
    // CHECK: }
    // CHECK: case b10: {
    // CHECK:   sv.bpassign %casez_tmp, %2 : i8
    // CHECK: }
    // CHECK: default: {
    // CHECK:   sv.bpassign %casez_tmp, %x_i8 : i8
    // CHECK: }

    // CHECK: %3 = sv.read_inout %casez_tmp : !hw.inout<i8>
    %2 = hw.array_get %muxed[%index_sel] : !hw.array<3xi8>, i2

    // CHECK: %4 = comb.icmp eq %3, %arg2 : i8
    // CHECK: sv.if %4  {
    %cond = comb.icmp eq %2, %arg2 : i8
    sv.if %cond {
      sv.fatal 1
    }
  }
}

// CHECK-LABEL: hw.module @array_create_concat_get_default
hw.module @array_create_concat_get_default(in %arg0: i8, in %arg1: i8, in %arg2: i8, in %arg3: i8,
                            in %sel: i2) {
  // CHECK: sv.initial  {
  sv.initial {
    // CHECK:   %casez_tmp = sv.reg : !hw.inout<i8>
    // CHECK:   %x_i8 = sv.constantX : i8
    // CHECK:   sv.case casez %sel : i2
    // CHECK:   case b00: {
    // CHECK:     sv.bpassign %casez_tmp, %arg0 : i8
    // CHECK:   }
    // CHECK:   case b01: {
    // CHECK:     sv.bpassign %casez_tmp, %arg1 : i8
    // CHECK:   }
    // CHECK:   case b10: {
    // CHECK:     sv.bpassign %casez_tmp, %arg2 : i8
    // CHECK:   }
    // CHECK:   default: {
    // CHECK:     sv.bpassign %casez_tmp, %x_i8 : i8
    // CHECK:   }
    %one_array = hw.array_create %arg2 : i8
    %two_array = hw.array_create %arg1, %arg0 : i8
    %three_array = hw.array_concat %one_array, %two_array : !hw.array<1xi8>, !hw.array<2xi8>

    // CHECK:   %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
    %2 = hw.array_get %three_array[%sel] : !hw.array<3xi8>, i2

    // CHECK:   %1 = comb.icmp eq %0, %arg2 : i8
    // CHECK:   sv.if %1  {
    %cond = comb.icmp eq %2, %arg2 : i8
    sv.if %cond {
      sv.fatal 1
    }
  }
}

// CHECK-LABEL: hw.module @array_constant_get_comb
hw.module @array_constant_get_comb(in %sel: i2, out a: i8) {
  // CHECK: %casez_tmp = sv.reg  : !hw.inout<i8>
  // CHECK: sv.alwayscomb  {
  // CHECK:   sv.case casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.bpassign %casez_tmp, %c3_i8 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %c2_i8 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %c1_i8 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %c0_i8 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.aggregate_constant [0 : i8, 1 : i8, 2 : i8, 3 : i8] : !hw.array<4xi8>
  // CHECK: %0 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>, i2

  // CHECK: hw.output %0 : i8
  hw.output %1 : i8
}

// CHECK-LABEL: hw.module @array_muxed_constant_get_comb
hw.module @array_muxed_constant_get_comb(in %array_sel: i1, in %index_sel: i2, out a: i8) {
  // CHECK: %0 = comb.mux %array_sel, %c3_i8, %c7_i8 : i8
  // CHECK: %1 = comb.mux %array_sel, %c2_i8, %c6_i8 : i8
  // CHECK: %2 = comb.mux %array_sel, %c1_i8, %c5_i8 : i8
  // CHECK: %3 = comb.mux %array_sel, %c0_i8, %c4_i8 : i8
  // CHECK: %casez_tmp = sv.reg : !hw.inout<i8>
  // CHECK: sv.alwayscomb {
  // CHECK:  sv.case casez %index_sel : i2
  // CHECK:  case b00: {
  // CHECK:    sv.bpassign %casez_tmp, %0 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %1 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %2 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %3 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.aggregate_constant [0 : i8, 1 : i8, 2 : i8, 3 : i8] : !hw.array<4xi8>
  %1 = hw.aggregate_constant [4 : i8, 5 : i8, 6 : i8, 7 : i8] : !hw.array<4xi8>
  %muxed = comb.mux %array_sel, %0, %1 : !hw.array<4xi8>
  // CHECK: %4 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %3 = hw.array_get %muxed[%index_sel] : !hw.array<4xi8>, i2

  // CHECK: hw.output %4 : i8
  hw.output %3 : i8
}

// CHECK-LABEL: hw.module @array_reg_mux_2
hw.module @array_reg_mux_2(in %clock: i1, in %arg0: i8, in %arg1: i8, in %sel: i1, out a: i8) {
  // CHECK: %reg = sv.reg : !hw.inout<i8>
  // CHECK: %reg_0 = sv.reg name "reg" : !hw.inout<i8>
  %reg = sv.reg : !hw.inout<array<2xi8>>
  // CHECK: sv.alwaysff(posedge %clock) {
  sv.alwaysff(posedge %clock)  {
    // CHECK: sv.passign %reg, %arg1 : i8
    // CHECK: sv.passign %reg_0, %arg0 : i8
    %0 = hw.array_create %arg0, %arg1 : i8
    sv.passign %reg, %0 : !hw.array<2xi8>
  // CHECK: }
  }

  // CHECK: %0 = sv.read_inout %reg : !hw.inout<i8>
  // CHECK: %1 = sv.read_inout %reg_0 : !hw.inout<i8>
  // CHECK: %casez_tmp = sv.reg : !hw.inout<i8>
  // CHECK: sv.alwayscomb {
  // CHECK:   sv.case casez %sel : i1
  // CHECK:   case b0: {
  // CHECK:     sv.bpassign %casez_tmp, %0 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %1 : i8
  // CHECK:   }
  // CHECK: }
  %1 = sv.array_index_inout %reg[%sel] : !hw.inout<array<2xi8>>, i1
  // CHECK: %2 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %2 = sv.read_inout %1 : !hw.inout<i8>
  // CHECK: hw.output %2 : i8
  hw.output %2 : i8
}

// CHECK-LABEL: hw.module @array_reg_mux_4
hw.module @array_reg_mux_4(in %arg0: i8, in %arg1: i8, in %arg2: i8,
                           in %arg3: i8, in %sel: i2, in %clock: i1,
                           out a: i8) {
  // CHECK: %reg = sv.reg : !hw.inout<i8>
  // CHECK: %reg_0 = sv.reg name "reg" : !hw.inout<i8>
  // CHECK: %reg_1 = sv.reg name "reg" : !hw.inout<i8>
  // CHECK: %reg_2 = sv.reg name "reg" : !hw.inout<i8>
  %reg = sv.reg : !hw.inout<array<4xi8>>
  // CHECK: sv.alwaysff(posedge %clock) {
  sv.alwaysff(posedge %clock)  {
    // CHECK: sv.passign %reg, %arg3 : i8
    // CHECK: sv.passign %reg_0, %arg2 : i8
    // CHECK: sv.passign %reg_1, %arg1 : i8
    // CHECK: sv.passign %reg_2, %arg0 : i8
    %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8
    sv.passign %reg, %0 : !hw.array<4xi8>
  // CHECK: }
  }
  // CHECK: %0 = sv.read_inout %reg : !hw.inout<i8>
  // CHECK: %1 = sv.read_inout %reg_0 : !hw.inout<i8>
  // CHECK: %2 = sv.read_inout %reg_1 : !hw.inout<i8>
  // CHECK: %3 = sv.read_inout %reg_2 : !hw.inout<i8>
  // CHECK: %casez_tmp = sv.reg : !hw.inout<i8>
  // CHECK: sv.alwayscomb {
  // CHECK:   sv.case casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.bpassign %casez_tmp, %0 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.bpassign %casez_tmp, %1 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.bpassign %casez_tmp, %2 : i8
  // CHECK:   }
  // CHECK:   default: {
  // CHECK:     sv.bpassign %casez_tmp, %3 : i8
  // CHECK:   }
  // CHECK: }
  %1 = sv.array_index_inout %reg[%sel] : !hw.inout<array<4xi8>>, i2
  // CHECK: %4 = sv.read_inout %casez_tmp : !hw.inout<i8>
  %2 = sv.read_inout %1 : !hw.inout<i8>
  // CHECK: hw.output %4 : i8
  hw.output %2 : i8
}

}  // end builtin.module
