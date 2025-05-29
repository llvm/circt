hw.module @Arrays(in %u0: i42, in %u1: i42, in %u2: i42, in %u3: i20, in %u4: i22) {
  %c0_i2 = hw.constant 0 : i2
  %c1_i2 = hw.constant 1 : i2
  %false = hw.constant false
  %true = hw.constant true
  %c0_i6 = hw.constant 0 : i6
  %c20_i6 = hw.constant 20 : i6
  %0 = builtin.unrealized_conversion_cast to !hw.struct<a: !hw.array<3xi42>, b: i42>
  %1 = llhd.constant_time <0ns, 0d, 1e>
  %s = llhd.sig %0 : !hw.struct<a: !hw.array<3xi42>, b: i42>
  %a = llhd.sig.struct_extract %s["a"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %a12 = llhd.sig.array_slice %a at %c1_i2 : (!hw.inout<array<3xi42>>) -> !hw.inout<array<2xi42>>
  %a0 = llhd.sig.array_get %a[%c0_i2] : !hw.inout<array<3xi42>>
  %a1 = llhd.sig.array_get %a12[%false] : !hw.inout<array<2xi42>>
  %a2 = llhd.sig.array_get %a12[%true] : !hw.inout<array<2xi42>>
  llhd.drv %a0, %u0 after %1 : !hw.inout<i42>
  llhd.drv %a1, %u1 after %1 : !hw.inout<i42>
  llhd.drv %a2, %u2 after %1 : !hw.inout<i42>
  %bX = llhd.sig.struct_extract %s["b"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %bY = llhd.sig.struct_extract %s["b"] : !hw.inout<struct<a: !hw.array<3xi42>, b: i42>>
  %b0 = llhd.sig.extract %bX from %c0_i6 : (!hw.inout<i42>) -> !hw.inout<i20>
  %b1 = llhd.sig.extract %bY from %c20_i6 : (!hw.inout<i42>) -> !hw.inout<i22>
  llhd.drv %b0, %u3 after %1 : !hw.inout<i20>
  llhd.drv %b1, %u4 after %1 : !hw.inout<i22>
}
