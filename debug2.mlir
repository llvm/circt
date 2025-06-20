hw.module @Foo(in %u: !hw.array<4xi42>, in %v: i42, in %w: i42, in %i: i2, in %q: i1, in %r: i1, in %s: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : !hw.array<4xi42>
  llhd.process {
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    llhd.drv %a, %u after %0 : !hw.inout<array<4xi42>>
    llhd.drv %1, %v after %0 if %q : !hw.inout<i42>
    llhd.drv %1, %w after %0 if %r : !hw.inout<i42>
    %2 = llhd.prb %a : !hw.inout<array<4xi42>>
    %3 = llhd.prb %1 : !hw.inout<i42>
    func.call @use_array_i42(%2) : (!hw.array<4xi42>) -> ()
    func.call @use_i42(%3) : (i42) -> ()
    llhd.halt
  }
  llhd.process {
    %1 = llhd.sig.array_get %a[%i] : !hw.inout<array<4xi42>>
    llhd.drv %a, %u after %0 if %q : !hw.inout<array<4xi42>>
    llhd.drv %1, %v after %0 if %r : !hw.inout<i42>
    llhd.drv %1, %w after %0 if %s : !hw.inout<i42>
    %2 = llhd.prb %a : !hw.inout<array<4xi42>>
    %3 = llhd.prb %1 : !hw.inout<i42>
    func.call @use_array_i42(%2) : (!hw.array<4xi42>) -> ()
    func.call @use_i42(%3) : (i42) -> ()
    llhd.halt
  }
}

func.func private @use_i42(%arg0: i42)
func.func private @use_array_i42(%arg0: !hw.array<4xi42>)
