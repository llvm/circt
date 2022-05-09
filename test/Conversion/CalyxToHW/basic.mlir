// RUN: circt-opt %s -lower-calyx-to-hw | FileCheck %s

// Sample program:
//
// component main(a: 32, b: 32, c: 32) -> (out: 32) {
//   cells {
//     mul = std_mult_pipe(32);
//     add = std_add(32);
//     buf = std_reg(32);
//   }
//
//   wires {
//     out = buf.out;
//     group do_mul {
//       mul.left = a;
//       mul.right= b;
//       mul.go = !mul.done ? 1'd1;
//       do_mul[done] = mul.done;
//     }
//     group do_add {
//       add.left = mul.out;
//       add.right = c;
//       buf.in = add.out;
//       buf.write_en = 1'd1;
//       do_add[done] = buf.done;
//     }
//   }
//
//   control {
//     seq {
//       do_mul;
//       do_add;
//     }
//   }
// }
//
// Compiled with:
//
// futil -p pre-opt -p compile -p post-opt -p lower -p lower-guards -b mlir

calyx.program "main" {
  calyx.component @main(%a: i32, %b: i32, %c: i32, %go: i1 {go = 1 : i64}, %clk: i1 {clk = 1 : i64}, %reset: i1 {reset = 1 : i64}) -> (%out: i32, %done: i1 {done = 1 : i64}) {
    %mul.clk, %mul.reset, %mul.go, %mul.left, %mul.right, %mul.out, %mul.done = calyx.std_mult_pipe @mul : i1, i1, i1, i32, i32, i32, i1
    %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32
    %buf.in, %buf.write_en, %buf.clk, %buf.reset, %buf.out, %buf.done = calyx.register @buf : i32, i1, i1, i1, i32, i1
    %true = hw.constant true
    %fsm.in, %fsm.write_en, %fsm.clk, %fsm.reset, %fsm.out, %fsm.done = calyx.register @fsm {generated = 1 : i64} : i2, i1, i1, i1, i2, i1
    %c-2_i2 = hw.constant -2 : i2
    %c0_i2 = hw.constant 0 : i2
    %c1_i2 = hw.constant 1 : i2
    // CHECK-DAG: do_mul_go = sv.wire
    %do_mul_go.in, %do_mul_go.out = calyx.std_wire @do_mul_go {generated = 1 : i64} : i1, i1
    // CHECK-DAG: do_mul_done = sv.wire
    %do_mul_done.in, %do_mul_done.out = calyx.std_wire @do_mul_done {generated = 1 : i64} : i1, i1
    // CHECK-DAG: do_add_go = sv.wire
    %do_add_go.in, %do_add_go.out = calyx.std_wire @do_add_go {generated = 1 : i64} : i1, i1
    // CHECK-DAG: do_add_done = sv.wire
    %do_add_done.in, %do_add_done.out = calyx.std_wire @do_add_done {generated = 1 : i64} : i1, i1
    // CHECK-DAG: tdcc_go = sv.wire
    %tdcc_go.in, %tdcc_go.out = calyx.std_wire @tdcc_go {generated = 1 : i64} : i1, i1
    // CHECK-DAG: tdcc_done = sv.wire
    %tdcc_done.in, %tdcc_done.out = calyx.std_wire @tdcc_done {generated = 1 : i64} : i1, i1
    // CHECK-DAG: comb.xor
    %not0.in, %not0.out = calyx.std_not @not0 {generated = 1 : i64} : i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq.left, %eq.right, %eq.out = calyx.std_eq @eq {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and0.left, %and0.right, %and0.out = calyx.std_and @and0 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and1.left, %and1.right, %and1.out = calyx.std_and @and1 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.xor
    %not1.in, %not1.out = calyx.std_not @not1 {generated = 1 : i64} : i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq0.left, %eq0.right, %eq0.out = calyx.std_eq @eq0 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and2.left, %and2.right, %and2.out = calyx.std_and @and2 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and3.left, %and3.right, %and3.out = calyx.std_and @and3 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq1.left, %eq1.right, %eq1.out = calyx.std_eq @eq1 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.icmp eq
    %eq2.left, %eq2.right, %eq2.out = calyx.std_eq @eq2 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and4.left, %and4.right, %and4.out = calyx.std_and @and4 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and5.left, %and5.right, %and5.out = calyx.std_and @and5 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq3.left, %eq3.right, %eq3.out = calyx.std_eq @eq3 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and6.left, %and6.right, %and6.out = calyx.std_and @and6 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and7.left, %and7.right, %and7.out = calyx.std_and @and7 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq4.left, %eq4.right, %eq4.out = calyx.std_eq @eq4 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.icmp eq
    %eq5.left, %eq5.right, %eq5.out = calyx.std_eq @eq5 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and8.left, %and8.right, %and8.out = calyx.std_and @and8 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and9.left, %and9.right, %and9.out = calyx.std_and @and9 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.or
    %or0.left, %or0.right, %or0.out = calyx.std_or @or0 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq6.left, %eq6.right, %eq6.out = calyx.std_eq @eq6 {generated = 1 : i64} : i2, i2, i1
    // CHECK-DAG: comb.and
    %and10.left, %and10.right, %and10.out = calyx.std_and @and10 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.and
    %and11.left, %and11.right, %and11.out = calyx.std_and @and11 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.or
    %or1.left, %or1.right, %or1.out = calyx.std_or @or1 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.xor
    %not2.in, %not2.out = calyx.std_not @not2 {generated = 1 : i64} : i1, i1
    // CHECK-DAG: comb.and
    %and12.left, %and12.right, %and12.out = calyx.std_and @and12 {generated = 1 : i64} : i1, i1, i1
    // CHECK-DAG: comb.icmp eq
    %eq7.left, %eq7.right, %eq7.out = calyx.std_eq @eq7 {generated = 1 : i64} : i2, i2, i1
    calyx.wires {
      // CHECK-DAG: sv.assign
      calyx.assign %done = %tdcc_done.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %out = %true ? %buf.out : i32
      // CHECK-DAG: sv.assign
      calyx.assign %add.left = %do_add_go.out ? %mul.out : i32
      // CHECK-DAG: sv.assign
      calyx.assign %add.right = %do_add_go.out ? %c : i32
      // CHECK-DAG: sv.assign
      calyx.assign %buf.clk = %true ? %clk : i1
      // CHECK-DAG: sv.assign
      calyx.assign %buf.in = %do_add_go.out ? %add.out : i32
      // CHECK-DAG: sv.assign
      calyx.assign %buf.reset = %true ? %reset : i1
      // CHECK-DAG: sv.assign
      calyx.assign %buf.write_en = %do_add_go.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %do_add_done.in = %true ? %buf.done : i1
      // CHECK-DAG: sv.assign
      calyx.assign %not0.in = %do_add_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %eq.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq.right = %c1_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and0.left = %not0.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and0.right = %eq.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and1.left = %and0.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and1.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %do_add_go.in = %and1.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %do_mul_done.in = %true ? %mul.done : i1
      // CHECK-DAG: sv.assign
      calyx.assign %not1.in = %do_mul_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %eq0.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq0.right = %c0_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and2.left = %not1.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and2.right = %eq0.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and3.left = %and2.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and3.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %do_mul_go.in = %and3.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.clk = %true ? %clk : i1
      // CHECK-DAG: sv.assign
      calyx.assign %eq1.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq1.right = %c-2_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.in = %eq1.out ? %c0_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq2.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq2.right = %c0_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and4.left = %eq2.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and4.right = %do_mul_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and5.left = %and4.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and5.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.in = %and5.out ? %c1_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq3.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq3.right = %c1_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and6.left = %eq3.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and6.right = %do_add_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and7.left = %and6.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and7.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.in = %and7.out ? %c-2_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.reset = %true ? %reset : i1
      // CHECK-DAG: sv.assign
      calyx.assign %eq4.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq4.right = %c-2_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq5.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq5.right = %c0_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and8.left = %eq5.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and8.right = %do_mul_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and9.left = %and8.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and9.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %or0.left = %eq4.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %or0.right = %and9.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %eq6.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq6.right = %c1_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %and10.left = %eq6.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and10.right = %do_add_done.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and11.left = %and10.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and11.right = %tdcc_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %or1.left = %or0.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %or1.right = %and11.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %fsm.write_en = %or1.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %mul.clk = %true ? %clk : i1
      // CHECK-DAG: sv.assign
      calyx.assign %not2.in = %mul.done : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and12.left = %not2.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %and12.right = %do_mul_go.out : i1
      // CHECK-DAG: sv.assign
      calyx.assign %mul.go = %and12.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %mul.left = %do_mul_go.out ? %a : i32
      // CHECK-DAG: sv.assign
      calyx.assign %mul.reset = %true ? %reset : i1
      // CHECK-DAG: sv.assign
      calyx.assign %mul.right = %do_mul_go.out ? %b : i32
      // CHECK-DAG: sv.assign
      calyx.assign %eq7.left = %fsm.out : i2
      // CHECK-DAG: sv.assign
      calyx.assign %eq7.right = %c-2_i2 : i2
      // CHECK-DAG: sv.assign
      calyx.assign %tdcc_done.in = %eq7.out ? %true : i1
      // CHECK-DAG: sv.assign
      calyx.assign %tdcc_go.in = %true ? %go : i1
    }
    calyx.control {
    }
  }
}
