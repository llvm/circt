// RUN: circt-opt -hw-legalize-modules -verify-diagnostics %s | FileCheck %s

module attributes {circt.loweringOptions = "disallowClockedAssertions"} {

hw.module @clocked_assert(in %clock : i1, in %prop : i1) {
  sv.assert_property %prop on posedge %clock : i1
}

// CHECK:      hw.module @clocked_assert(in %clock : i1, in %prop : i1) {
// CHECK-NEXT:   sv.always posedge %clock {
// CHECK-NEXT:     sv.assert_property %prop : i1
// CHECK-NEXT:   }

hw.module @clocked_assume(in %clock : i1, in %prop : i1) {
  sv.assume_property %prop on posedge %clock : i1
}

// CHECK:      hw.module @clocked_assume(in %clock : i1, in %prop : i1) {
// CHECK-NEXT:   sv.always posedge %clock {
// CHECK-NEXT:     sv.assume_property %prop : i1
// CHECK-NEXT:   }

hw.module @clocked_cover(in %clock : i1, in %prop : i1) {
  sv.cover_property %prop on posedge %clock : i1
}

// CHECK:      hw.module @clocked_cover(in %clock : i1, in %prop : i1) {
// CHECK-NEXT:   sv.always posedge %clock {
// CHECK-NEXT:     sv.cover_property %prop : i1
// CHECK-NEXT:   }

} // end builtin.module
