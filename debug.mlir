// CHECK-LABEL: @BasicSigExtract
hw.module @BasicSigExtract(in %u: i42, in %v: i10, in %i: i6, in %q: i1) {
  %0 = llhd.constant_time <0ns, 0d, 1e>
  %a = llhd.sig %u : i42
  // CHECK: llhd.process
  llhd.process {
    // CHECK-NOT: llhd.drv
    llhd.drv %a, %u after %0 : !hw.inout<i42>
    // CHECK-NOT: llhd.sig.extract
    %1 = llhd.sig.extract %a from %i : (!hw.inout<i42>) -> !hw.inout<i10>
    // CHECK-NOT: llhd.drv
    // CHECK-NEXT: [[A:%.+]] = hw.array_inject %u[%i], %v
    llhd.drv %1, %v after %0 if %q : !hw.inout<i10>
    // CHECK-NOT: llhd.prb
    %2 = llhd.prb %a : !hw.inout<i42>
    // CHECK-NEXT: call @use_i42([[A]])
    func.call @use_i42(%2) : (i42) -> ()
    // CHECK-NEXT: llhd.constant_time
    // CHECK-NEXT: llhd.drv %a, [[A]]
    // CHECK-NEXT: llhd.halt
    llhd.halt
  }
}

func.func private @use_i42(%arg0: i42)
