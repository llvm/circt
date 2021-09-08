//RUN: circt-translate --export-llhd-verilog -split-input-file -verify-diagnostics %s

llhd.entity @check_invalid_drv_time () -> () {
  %0 = hw.constant 1 : i1
  %1 = llhd.sig "sigI1" %0 : i1
  // expected-error @+2 {{Not possible to translate a time attribute with 0 real time and non-1 delta.}}
  // expected-error @+1 {{Operation not supported!}}
  %2 = llhd.constant_time #llhd.time<0ns, 0d, 0e>
  llhd.drv %1, %0 after %2 : !llhd.sig<i1>
}
