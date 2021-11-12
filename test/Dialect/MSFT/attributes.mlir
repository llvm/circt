// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: msft.physical_region @region1
msft.physical_region @region1, [
  // CHECK-SAME: #msft.physical_bounds<x: [0, 10], y: [0, 10]>
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  // CHECK-SAME: #msft.physical_bounds<x: [20, 30], y: [20, 30]>
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]

msft.module.extern @ext()

msft.module @mod {} () -> () {
  msft.instance @inst @ext() {
    // CHECK: "msft:locate" = #msft.physical_region_ref<@region1>
    "msft:locate" = #msft.physical_region_ref<@region1>
  }: () -> ()
  msft.output
}
