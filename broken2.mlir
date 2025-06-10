hw.module @RegressionOverlappingDrives(in %u: i153, in %v: i5) {
  %c42_i8 = hw.constant 42 : i8
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i153
  %1 = llhd.sig.extract %a from %c42_i8 : (!hw.inout<i153>) -> !hw.inout<i5>
  llhd.drv %a, %u after %0 : !hw.inout<i153>
  llhd.drv %1, %v after %0 : !hw.inout<i5>
}
