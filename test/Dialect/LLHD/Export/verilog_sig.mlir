//RUN: circt-translate --llhd-to-verilog -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: _check_sig
llhd.entity @check_sig () -> () {
  // CHECK-NEXT: wire _[[A:.*]] = 1'd1;
  %0 = llhd.const 1 : i1
  // CHECK-NEXT: wire [63:0] _[[B:.*]] = 64'd256;
  %1 = llhd.const 256 : i64
  %2 = llhd.const #llhd.time<1ns, 0d, 0e> : !llhd.time
  // CHECK-NEXT: var _[[C:.*]] = _[[A]];
  %3 = llhd.sig "sigI1" %0 : i1
  // CHECK-NEXT: var [63:0] _{{.*}} = _[[B]];
  %4 = llhd.sig "sigI64" %1 : i64
  %5 = llhd.prb %3 : !llhd.sig<i1>
  // CHECK-NEXT: assign _[[C]] = #(1ns) _[[A]];
  llhd.drv %3, %0 after %2 : !llhd.sig<i1>
  %6 = llhd.const #llhd.time<0ns, 1d, 0e> : !llhd.time
  // CHECK-NEXT: assign _[[C]] = #(0ns) _[[A]];
  llhd.drv %3, %0 after %6 : !llhd.sig<i1>
  // CHECK-NEXT: assign _[[C]] = #(0ns) _[[A]] ? _[[A]] : _[[C]];
  llhd.drv %3, %0 after %6 if %0 : !llhd.sig<i1>
}

// -----

llhd.entity @check_invalid_drv_time () -> () {
  %0 = llhd.const 1 : i1
  %1 = llhd.sig "sigI1" %0 : i1
  // expected-error @+2 {{Not possible to translate a time attribute with 0 real time and non-1 delta.}}
  // expected-error @+1 {{Operation not supported!}}
  %2 = llhd.const #llhd.time<0ns, 0d, 0e> : !llhd.time
  llhd.drv %1, %0 after %2 : !llhd.sig<i1>
}
