// RUN: circt-translate --export-calyx --verify-diagnostics %s | FileCheck %s --strict-whitespace

// CHECK: import "primitives/core.futil";
calyx.program {
  // CHECK-LABEL: component A(in: 8, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out: 8, @done done: 1) {
  calyx.component @A(%in: i8, %go: i1 {go = 1}, %clk: i1 {clk = 1}, %reset: i1 {reset = 1}) -> (%out: i8, %done: i1 {done = 1}) {
    %c1_1 = hw.constant 1 : i1

    calyx.wires {
      // CHECK: done = 1'd1;
      calyx.assign %done = %c1_1 : i1
    }
    calyx.control {}
  }

  // CHECK-LABEL: component B(in: 1, @go go: 1, @clk clk: 1, @reset reset: 1) -> (out: 1, @done done: 1) {
  calyx.component @B(%in: i1, %go: i1 {go = 1}, %clk: i1 {clk = 1}, %reset: i1 {reset = 1}) -> (%out: i1, %done: i1 {done = 1}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }

  // CHECK-LABEL: component main(@go go: 1, @clk clk: 1, @reset reset: 1) -> (@done done: 1) {
  calyx.component @main(%go: i1 {go = 1}, %clk: i1 {clk = 1}, %reset: i1 {reset = 1}) -> (%done: i1 {done = 1}) {
    // CHECK-LABEL: cells {
    // CHECK-NEXT:    c0 = A();
    // CHECK-NEXT:    c1 = B();
    // CHECK-NEXT:    r = std_reg(8);
    // CHECK-NEXT:    m0 = std_mem_d1(32, 1, 1);
    // CHECK-NEXT:    m1 = std_mem_d2(8, 64, 64, 6, 6);
    // CHECK-NEXT:    a0 = std_add(32);
    // CHECK-NEXT:    s0 = std_slice(32, 8);
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance "c0" @A : i8, i1, i1, i1, i8, i1
    %c1.in, %c1.go, %c1.clk, %c1.reset, %c1.out, %c1.done = calyx.instance "c1" @B : i1, i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register "r" : i8, i1, i1, i1, i8, i1
    %m0.addr0, %m0.write_data, %m0.write_en, %m0.clk, %m0.read_data, %m0.done = calyx.memory "m0"<[1] x 32> [1] : i1, i32, i1, i1, i32, i1
    %m1.addr0, %m1.addr1, %m1.write_data, %m1.write_en, %m1.clk, %m1.read_data, %m1.done = calyx.memory "m1"<[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %a0.left, %a0.right, %a0.out = calyx.std_add "a0" : i32, i32, i32
    %s0.in, %s0.out = calyx.std_slice "s0" : i32, i8
    %c0 = hw.constant 0 : i1
    %c1 = hw.constant 1 : i1
    // CHECK-LABEL: wires {
    calyx.wires {
      // CHECK-NEXT: group Group1 {
      // CHECK-NEXT:    s0.in = a0.out;
      // CHECK-NEXT:    a0.left = m0.read_data;
      // CHECK-NEXT:    Group1[go] = 1'd0;
      // CHECK-NEXT:    c0.in = c0.out;
      // CHECK-NEXT:    Group1[done] = c0.done;
      calyx.group @Group1 {
        calyx.assign %s0.in = %a0.out : i32
        calyx.assign %a0.left = %m0.read_data : i32
        calyx.group_go %c0 : i1
        calyx.assign %c0.in = %c0.out : i8
        calyx.group_done %c0.done : i1
      }
      // CHECK-LABEL: comb group Group2 {
      // CHECK-NEXT:     c1.in = (c1.out | (c1.out & 1'd1 & !c1.out)) ? c1.out;
      calyx.comb_group @Group2 {
        %not = comb.xor %c1.out, %c1 : i1
        %and = comb.and %c1.out, %c1, %not : i1
        %or = comb.or %c1.out, %and : i1
        calyx.assign %c1.in = %c1.out, %or ? : i1
      }
      // CHECK-LABEL: group Group3 {
      // CHECK-NEXT:     r.in = c0.out;
      // CHECK-NEXT:     r.write_en = 1'd1;
      // CHECK-NEXT:     Group3[done] = r.done;
      calyx.group @Group3 {
        calyx.assign %r.in = %c0.out : i8
        calyx.assign %r.write_en = %c1 : i1
        calyx.group_done %r.done : i1
      }
      // CHECK:   c0.go = c1.out;
      calyx.assign %c0.go = %c1.out : i1
    }
    // CHECK-LABEL: control {
    // CHECK-NEXT:    seq {
    // CHECK-NEXT:      par {
    // CHECK-NEXT:        Group1;
    // CHECK-NEXT:        Group3;
    // CHECK-NEXT:      }
    // CHECK-NEXT:      seq {
    // CHECK-NEXT:        Group1;
    // CHECK-NEXT:        while c1.in with Group2 {
    // CHECK-NEXT:          seq {
    // CHECK-NEXT:            Group1;
    // CHECK-NEXT:            Group1;
    // CHECK-NEXT:            if c1.in with Group2 {
    // CHECK-NEXT:              Group1;
    // CHECK-NEXT:            }
    // CHECK-NEXT:            if c1.in {
    // CHECK-NEXT:              Group1;
    // CHECK-NEXT:            }
    // CHECK-NEXT:            while c1.in {
    // CHECK-NEXT:              Group1;
    // CHECK-NEXT:            }
    // CHECK-NEXT:          }
    // CHECK-NEXT:        }
    // CHECK-NEXT:      }
    // CHECK-NEXT:    }
    // CHECK-NEXT:  }
    calyx.control {
      calyx.seq {
        calyx.par {
          calyx.enable @Group1
          calyx.enable @Group3
        }
        calyx.seq {
          calyx.enable @Group1
          calyx.while %c1.in with @Group2 {
            calyx.seq {
              calyx.enable @Group1
              calyx.enable @Group1
              calyx.if %c1.in with @Group2 {
                calyx.enable @Group1
              }
              calyx.if %c1.in {
                calyx.enable @Group1
              }
              calyx.while %c1.in {
                calyx.enable @Group1
              }
            }
          }
        }
      }
    }
  }
}
