// RUN: circt-translate --export-calyx --verify-diagnostics %s
// TODO: Add CHECKs when Calyx control operations are supported in CalyxEmitter.

calyx.program {
  calyx.component @A(%in: i1, %go: i1, %clk: i1, %reset: i1) -> (%out: i1, %done: i1) {
    calyx.wires {}
    calyx.control {}
  }

  calyx.component @main(%in: i32, %go: i1, %clk: i1, %reset: i1) -> (%out: i32, %done: i1) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.cell "c0" @A : i1, i1, i1, i1, i1, i1
    
    calyx.wires {
      calyx.group @Group1 {
        %c1_i1 = hw.constant 1 : i1
        calyx.group_done %c1_i1 : i1 
      }
      calyx.group @Group2 {
        %c1_i1 = hw.constant 1 : i1
        calyx.group_done %c1_i1 : i1 
      }
      calyx.group @Group3 {
        %c1_i1 = hw.constant 1 : i1
        calyx.group_done %c1_i1 : i1 
      }
    }
    calyx.control {
      calyx.seq {
        calyx.if %c0.out, @Group1 {
          calyx.enable @Group2
        }
        calyx.if %c0.out, @Group1 {
          calyx.if %c0.out, @Group1 {
            calyx.enable @Group2
          }
          calyx.seq {
            calyx.if %c0.out, @Group1 {
              calyx.enable @Group2
            } else {
              calyx.enable @Group3
            }
          }
        }
      }
    }
  }
}
