// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: msft.physical_region @region1
msft.physical_region @region1, [
  // CHECK-SAME: #msft.physical_bounds<x: [0, 10], y: [0, 10]>
  #msft.physical_bounds<x: [0, 10], y: [0, 10]>,
  // CHECK-SAME: #msft.physical_bounds<x: [20, 30], y: [20, 30]>
  #msft.physical_bounds<x: [20, 30], y: [20, 30]>]
