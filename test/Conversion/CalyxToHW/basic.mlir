// RUN: circt-opt %s -lower-calyx-to-hw | FileCheck %s

// Sample program:
//
// component main(a: 32, b: 32) -> (out: 32) {
//   cells {
//     add = std_add(32);
//     buf = std_reg(32);
//   }
//
//   wires {
//     out = buf.out;
//     group g0 {
//       add.left = a;
//       add.right = b;
//       buf.in = add.out;
//       buf.write_en = 1'd1;
//       g0[done] = buf.done;
//     }
//   }
//
//   control {
//     g0;
//   }
// }
//
// Compiled with:
//
// futil -p pre-opt -p compile -p post-opt -p lower -p lower-guards -b mlir

calyx.program "main" {
  calyx.component @main(%a: i32, %b: i32, %go: i1 {go = 1 : i64}, %clk: i1 {clk = 1 : i64}, %reset: i1 {reset = 1 : i64}) -> (%out: i32, %done: i1 {done = 1 : i64}) {
    // CHECK-DAG:  %add_left = sv.wire  : !hw.inout<i32>
    // CHECK-DAG:  %2 = sv.read_inout %add_left : !hw.inout<i32>
    // CHECK-DAG:  %add_right = sv.wire  : !hw.inout<i32>
    // CHECK-DAG:  %3 = sv.read_inout %add_right : !hw.inout<i32>
    // CHECK-DAG:  %4 = comb.add %2, %3 : i32
    // CHECK-DAG:  %add_out = sv.wire  : !hw.inout<i32>
    // CHECK-DAG:  sv.assign %add_out, %4 : i32
    // CHECK-DAG:  %5 = sv.read_inout %add_out : !hw.inout<i32>
    %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32

    // CHECK-DAG:  %buf_in = sv.wire  : !hw.inout<i32>
    // CHECK-DAG:  %6 = sv.read_inout %buf_in : !hw.inout<i32>
    // CHECK-DAG:  %buf_write_en = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  %7 = sv.read_inout %buf_write_en : !hw.inout<i1>
    // CHECK-DAG:  %buf_clk = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  %8 = sv.read_inout %buf_clk : !hw.inout<i1>
    // CHECK-DAG:  %buf_reset = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  %9 = sv.read_inout %buf_reset : !hw.inout<i1>
    // CHECK-DAG:  %c0_i32 = hw.constant 0 : i32
    // CHECK-DAG:  %buf_reg = seq.compreg sym @buf_reg %6, %8, %9, %c0_i32  : i32
    // CHECK-DAG:  %false = hw.constant false
    // CHECK-DAG:  %buf_done_reg = seq.compreg sym @buf_done_reg %7, %8, %9, %false  : i1
    // CHECK-DAG:  %buf = sv.wire  : !hw.inout<i32>
    // CHECK-DAG:  sv.assign %buf, %buf_reg : i32
    // CHECK-DAG:  %10 = sv.read_inout %buf : !hw.inout<i32>
    // CHECK-DAG:  %buf_done = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  sv.assign %buf_done, %buf_done_reg : i1
    // CHECK-DAG:  %11 = sv.read_inout %buf_done : !hw.inout<i1>
    %buf.in, %buf.write_en, %buf.clk, %buf.reset, %buf.out, %buf.done = calyx.register @buf : i32, i1, i1, i1, i32, i1

    // CHECK-DAG:  %g0_go = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  %12 = sv.read_inout %g0_go : !hw.inout<i1>
    %g0_go.in, %g0_go.out = calyx.std_wire @g0_go {generated = 1 : i64} : i1, i1

    // CHECK-DAG:  %g0_done = sv.wire  : !hw.inout<i1>
    // CHECK-DAG:  %13 = sv.read_inout %g0_done : !hw.inout<i1>
    %g0_done.in, %g0_done.out = calyx.std_wire @g0_done {generated = 1 : i64} : i1, i1

    calyx.wires {
      // CHECK-DAG:  %true = hw.constant true
      %true = hw.constant true

      // CHECK-DAG:  %false_0 = hw.constant false
      // CHECK-DAG:  %14 = comb.mux %13, %true, %false_0 : i1
      // CHECK-DAG:  sv.assign %done, %14 : i1
      calyx.assign %done = %g0_done.out ? %true : i1

      // CHECK-DAG:  %c0_i32_1 = hw.constant 0 : i32
      // CHECK-DAG:  %15 = comb.mux %true, %10, %c0_i32_1 : i32
      // CHECK-DAG:  sv.assign %out, %15 : i32
      calyx.assign %out = %true ? %buf.out : i32

      // CHECK-DAG:  %c0_i32_2 = hw.constant 0 : i32
      // CHECK-DAG:  %16 = comb.mux %12, %a, %c0_i32_2 : i32
      // CHECK-DAG:  sv.assign %add_left, %16 : i32
      calyx.assign %add.left = %g0_go.out ? %a : i32

      // CHECK-DAG:  %c0_i32_3 = hw.constant 0 : i32
      // CHECK-DAG:  %17 = comb.mux %12, %b, %c0_i32_3 : i32
      // CHECK-DAG:  sv.assign %add_right, %17 : i32
      calyx.assign %add.right = %g0_go.out ? %b : i32

      // CHECK-DAG:  %false_4 = hw.constant false
      // CHECK-DAG:  %18 = comb.mux %true, %clk, %false_4 : i1
      // CHECK-DAG:  sv.assign %buf_clk, %18 : i1
      calyx.assign %buf.clk = %true ? %clk : i1

      // CHECK-DAG:  %c0_i32_5 = hw.constant 0 : i32
      // CHECK-DAG:  %19 = comb.mux %12, %5, %c0_i32_5 : i32
      // CHECK-DAG:  sv.assign %buf_in, %19 : i32
      calyx.assign %buf.in = %g0_go.out ? %add.out : i32

      // CHECK-DAG:  %false_6 = hw.constant false
      // CHECK-DAG:  %20 = comb.mux %true, %reset, %false_6 : i1
      // CHECK-DAG:  sv.assign %buf_reset, %20 : i1
      calyx.assign %buf.reset = %true ? %reset : i1

      // CHECK-DAG:  %false_7 = hw.constant false
      // CHECK-DAG:  %21 = comb.mux %12, %true, %false_7 : i1
      // CHECK-DAG:  sv.assign %buf_write_en, %21 : i1
      calyx.assign %buf.write_en = %g0_go.out ? %true : i1

      // CHECK-DAG:  %false_8 = hw.constant false
      // CHECK-DAG:  %22 = comb.mux %true, %11, %false_8 : i1
      // CHECK-DAG:  sv.assign %g0_done, %22 : i1
      calyx.assign %g0_done.in = %true ? %buf.done : i1

      // CHECK-DAG:  %false_9 = hw.constant false
      // CHECK-DAG:  %23 = comb.mux %true, %go, %false_9 : i1
      // CHECK-DAG:  sv.assign %g0_go, %23 : i1
      calyx.assign %g0_go.in = %true ? %go : i1

      // CHECK-DAG:  %out = sv.wire  : !hw.inout<i32>
      // CHECK-DAG:  %0 = sv.read_inout %out : !hw.inout<i32>
      // CHECK-DAG:  %done = sv.wire  : !hw.inout<i1>
      // CHECK-DAG:  %1 = sv.read_inout %done : !hw.inout<i1>
      // CHECK-DAG:  hw.output %0, %1 : i32, i1
    }
    calyx.control {
    }
  }
}
