// RUN: circt-opt %s -verify-diagnostics | circt-opt -verify-diagnostics | FileCheck %s

%0 = hw.constant 1 : i1

// CHECK: #msft.logic_locked_region<region1, 0, 10, 0, 10>
%1 = comb.add %0, %0 { region = #msft.logic_locked_region<region1, 0, 10, 0, 10> } : i1

// CHECK: #msft.logic_locked_region<region2, 0, 10, 0, 10>
%2 = comb.add %0, %0 { region = #msft.logic_locked_region<region2, 0, 10, 0, 10> } : i1
