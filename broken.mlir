// circt-verilog broken.mlir
hw.module @Foo() {
  %c0_i5 = hw.constant 0 : i5
  %c-118_i8 = hw.constant -118 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %c0_i305 = hw.constant 0 : i305
  %c0_i153 = hw.constant 0 : i153
  %a = llhd.sig %c0_i153 : i153
  llhd.process {
    cf.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb1
    llhd.drv %a, %c0_i153 after %0 : !hw.inout<i153>
    %1 = llhd.sig.extract %a from %c-118_i8 : (!hw.inout<i153>) -> !hw.inout<i5>
    llhd.drv %1, %c0_i5 after %0 : !hw.inout<i5>
    llhd.wait (%c0_i153, %c0_i305 : i153, i305), ^bb1
  }
}
