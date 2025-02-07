hw.module @Foo(in %u: i32, in %v: i32, in %bool: i1) {
  %epsilon = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %a = llhd.sig %c0_i32 : i32
  %b = llhd.sig %c0_i32 : i32

  // llhd.process {
  //   cf.br ^bb1
  // ^bb1:
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   cf.br ^bb2
  // ^bb2:
  //   %0 = llhd.prb %a : !hw.inout<i32>
  //   func.call @use_i32(%0) : (i32) -> ()
  //   cf.br ^bb3
  // ^bb3:
  //   llhd.halt
  // }

  // llhd.process {
  //   cf.cond_br %bool, ^bb1, ^bb3
  // ^bb1:
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   cf.br ^bb2
  // ^bb2:
  //   %0 = llhd.prb %a : !hw.inout<i32>
  //   func.call @use_i32(%0) : (i32) -> ()
  //   cf.br ^bb3
  // ^bb3:
  //   llhd.halt
  // }

  // // Check that additional basic blocks get inserted to accommodate probes after
  // // wait.
  // llhd.process {
  //   cf.br ^bb1
  // ^bb1:
  //   llhd.prb %a : !hw.inout<i32>
  //   llhd.wait ^bb1
  // }
  // llhd.process {
  //   %0 = hw.constant 0 : i42
  //   cf.br ^bb1(%0 : i42)
  // ^bb1(%1: i42):
  //   llhd.prb %a : !hw.inout<i32>
  //   llhd.wait ^bb1(%1 : i42)
  // }

  // // Check that additional basic blocks get inserted to accomodate drives when
  // // control flow diverges and reaching definitions cannot continue into all
  // // successors.
  // llhd.process {
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   cf.cond_br %bool, ^bb2, ^bb3
  // ^bb1:
  //   cf.br ^bb2
  // ^bb2:
  //   llhd.halt
  // ^bb3:
  //   llhd.halt
  // }

  // // Definitions from drives must not propagate across wait points.
  // llhd.process {
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   llhd.wait ^bb1
  // ^bb1:
  //   llhd.halt
  // }

  llhd.process {
    llhd.drv %a, %u after %epsilon : !hw.inout<i32>
    llhd.drv %b, %u after %epsilon : !hw.inout<i32>
    cf.br ^bb2
  ^bb1:
    llhd.drv %a, %v after %epsilon : !hw.inout<i32>
    llhd.drv %b, %v after %epsilon : !hw.inout<i32>
    cf.br ^bb2
  ^bb2:
    %0 = llhd.prb %a : !hw.inout<i32>
    func.call @use_i32(%0) : (i32) -> ()
    cf.br ^bb3
  ^bb3:
    llhd.halt
  }

  // // This process requires an auxiliary block to be inserted between bb0 and bb2
  // // to materialize a probe, since bb2 can also be reached without the
  // // definition.
  // llhd.process {
  //   cf.cond_br %bool, ^bb1, ^bb2
  // ^bb1:
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   cf.br ^bb2
  // ^bb2:
  //   %0 = llhd.prb %a : !hw.inout<i32>
  //   llhd.halt
  // }

  // // This process requires an auxiliary block to be inserted between bb1 and bb2
  // // to materialize the drive, since bb2 can also be reached without the
  // // definition.
  // llhd.process {
  //   cf.cond_br %bool, ^bb1, ^bb2
  // ^bb1:
  //   llhd.drv %a, %u after %epsilon : !hw.inout<i32>
  //   cf.cond_br %bool, ^bb2, ^bb3
  // ^bb2:
  //   llhd.halt
  // ^bb3:
  //   llhd.halt
  // }
}

func.func private @use_i32(%arg0: i32)
