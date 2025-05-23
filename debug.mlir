hw.module @Arrays(in %u0: i42, in %u1: i42, in %u2: i42) {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %c2_i2 = hw.constant 2 : i2
  %0 = builtin.unrealized_conversion_cast to !hw.array<3xi42>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %0 : !hw.array<3xi42>
  %a0 = llhd.sig.array_get %a[%c0_i2] : !hw.inout<array<3xi42>>
  %a1 = llhd.sig.array_get %a[%c1_i2] : !hw.inout<array<3xi42>>
  %a2 = llhd.sig.array_get %a[%c2_i2] : !hw.inout<array<3xi42>>
  llhd.drv %a0, %u0 after %1 : !hw.inout<i42>
  llhd.drv %a1, %u1 after %1 : !hw.inout<i42>
  llhd.drv %a2, %u2 after %1 : !hw.inout<i42>
}
