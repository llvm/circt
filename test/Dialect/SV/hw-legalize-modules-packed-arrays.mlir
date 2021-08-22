// RUN: circt-opt -hw-legalize-modules -verify-diagnostics %s | FileCheck %s

module attributes {circt.loweringOptions = "disallowPackedArrays"} {

// CHECK-LABEL: hw.module @reject_arrays
hw.module @reject_arrays(%arg0: i8, %arg1: i8, %arg2: i8,
                         %arg3: i8, %sel: i2, %clock: i1)
   -> (%a: !hw.array<4xi8>) {
  // This needs full-on "legalize types" for the HW dialect.
  
  %reg = sv.reg  : !hw.inout<array<4xi8>>
  sv.alwaysff(posedge %clock)  {
    // expected-error @+1 {{unsupported packed array expression}}
    %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8
    sv.passign %reg, %0 : !hw.array<4xi8>
  }

  // expected-error @+1 {{unsupported packed array expression}}
  %1 = sv.read_inout %reg : !hw.inout<array<4xi8>>
  hw.output %1 : !hw.array<4xi8>
}

// CHECK-LABEL: hw.module @array_create_get_comb
hw.module @array_create_get_comb(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8,
                                 %sel: i2)
   -> (%a: i8) {
  // CHECK: %0 = sv.wire  : !hw.inout<i8>
  // CHECK: sv.alwayscomb  {
  // CHECK:   sv.casez %sel : i2
  // CHECK:   case b00: {
  // CHECK:     sv.passign %0, %arg0 : i8
  // CHECK:   }
  // CHECK:   case b01: {
  // CHECK:     sv.passign %0, %arg1 : i8
  // CHECK:   }
  // CHECK:   case b10: {
  // CHECK:     sv.passign %0, %arg2 : i8
  // CHECK:   }
  // CHECK:   case b11: {
  // CHECK:     sv.passign %0, %arg3 : i8
  // CHECK:   }
  // CHECK: }
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8

  // CHECK: %1 = sv.read_inout %0 : !hw.inout<i8>
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>

  // CHECK: hw.output %1 : i8
  hw.output %1 : i8
}

// CHECK-LABEL: hw.module @array_create_get_default
hw.module @array_create_get_default(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8,
                            %sel: i2) {
  // CHECK: %0 = sv.wire  : !hw.inout<i8>
  // CHECK: sv.initial  {
  sv.initial {
    // CHECK:   %x_i8 = sv.constantX : i8
    // CHECK:   sv.casez %sel : i2
    // CHECK:   case b00: {
    // CHECK:     sv.passign %0, %arg0 : i8
    // CHECK:   }
    // CHECK:   case b01: {
    // CHECK:     sv.passign %0, %arg1 : i8
    // CHECK:   }
    // CHECK:   case b10: {
    // CHECK:     sv.passign %0, %arg2 : i8
    // CHECK:   }
    // CHECK:   default: {
    // CHECK:     sv.passign %0, %x_i8 : i8
    // CHECK:   }
    %three_array = hw.array_create %arg0, %arg1, %arg2 : i8

    // CHECK:   %1 = sv.read_inout %0 : !hw.inout<i8>
    %2 = hw.array_get %three_array[%sel] : !hw.array<3xi8>

    // CHECK:   %2 = comb.icmp eq %1, %arg2 : i8
    // CHECK:   sv.if %2  {
    %cond = comb.icmp eq %2, %arg2 : i8
    sv.if %cond {
      sv.fatal
    }
  }
}

}  // end builtin.module