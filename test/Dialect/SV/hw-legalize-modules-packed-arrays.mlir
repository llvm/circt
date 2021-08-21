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

// CHECK-LABEL: hw.module @array_create_get
hw.module @array_create_get(%arg0: i8, %arg1: i8, %arg2: i8, %arg3: i8, %sel: i2)
   -> (%a: i8) {
  // expected-error @+1 {{unsupported packed array expression}}
  %0 = hw.array_create %arg0, %arg1, %arg2, %arg3 : i8
  %1 = hw.array_get %0[%sel] : !hw.array<4xi8>
  hw.output %1 : i8
}

}  // end builtin.module