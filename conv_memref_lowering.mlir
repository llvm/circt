module  {
  hir.func @convolution at %arg6 (%arg0 : !hir.bus<wr i1, wr tuple<i4, i4>, proto {join = "join_valid"}>, %arg1 : !hir.bus<rd i1, rd tuple<i4, i4>>, %arg2 : !hir.bus<wr i32>, %arg3 : !hir.bus<rd i32>, %arg4 : !hir.bus<wr i1, wr tuple<tuple<i4, i4>, i32>, proto {join = "join_valid"}>, %arg5 : !hir.bus<rd i1, rd tuple<tuple<i4, i4>, i32>>) {
    %0:2 = hir.alloca("bus") : tensor<3x!hir.bus<wr i1, wr tuple<i4>, proto {join = "join_valid"}>>, tensor<3x!hir.bus<rd i1, rd tuple<i4>>>
    %1:2 = hir.alloca("bus") : tensor<3x!hir.bus<wr i32>>, tensor<3x!hir.bus<rd i32>>
    %2:2 = hir.alloca("bus") : tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>, tensor<3x!hir.bus<rd i1, rd tuple<tuple<i4>, i32>>>
    hir.call @bram(%0#1, %1#0, %2#1) at %arg6 : !hir.func<(tensor<3x!hir.bus<rd i1, rd tuple<i4>>>, tensor<3x!hir.bus<wr i32>>, tensor<3x!hir.bus<rd i1, rd tuple<tuple<i4>, i32>>>) -> ()>
    %3:2 = hir.alloca("bus") : tensor<3x3x!hir.bus<wr i1, wr tuple<>, proto {join = "join_valid"}>>, tensor<3x3x!hir.bus<rd i1, rd tuple<>>>
    %4:2 = hir.alloca("bus") : tensor<3x3x!hir.bus<wr i32>>, tensor<3x3x!hir.bus<rd i32>>
    %5:2 = hir.alloca("bus") : tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>, tensor<3x3x!hir.bus<rd i1, rd tuple<tuple<>, i32>>>
    hir.call @bram(%3#1, %4#0, %5#1) at %arg6 : !hir.func<(tensor<3x3x!hir.bus<rd i1, rd tuple<>>>, tensor<3x3x!hir.bus<wr i32>>, tensor<3x3x!hir.bus<rd i1, rd tuple<tuple<>, i32>>>) -> ()>
    %6:2 = hir.alloca("bram") : !hir.memref<3x3xi32, [0, 1], {rd = 1 : i64}>, !hir.memref<3x3xi32, [0, 1], {wr = 1 : i64}>
    %7 = hir.constant(0 : i64) : !hir.const
    %8 = hir.constant(1 : i64) : !hir.const
    %9 = hir.constant(2 : i64) : !hir.const
    %10 = hir.constant(4 : i64) : !hir.const
    %11 = hir.constant(16 : i64) : !hir.const
    %12 = hir.for %arg7 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg8 = %arg6 + %8) {
      %15 = hir.for %arg9 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg10 = %arg8 + %8) {
        %17 = hir.delay %arg10 by %8 at %arg10 : !hir.time -> !hir.time
        hir.yield at %17
        hir.send %8 to %arg0[%7] at %arg10 : !hir.const to !hir.bus<wr i1, wr tuple<i4, i4>, proto {join = "join_valid"}>[!hir.const]
        %18 = hir.cast %arg7 : i32 -> i4
        %19 = hir.cast %arg9 : i32 -> i4
        %20 = hir.tuple(%18, %19) : (i4, i4) -> tuple<i4, i4>
        hir.send %20 to %arg0[%8] at %arg10 : tuple<i4, i4> to !hir.bus<wr i1, wr tuple<i4, i4>, proto {join = "join_valid"}>[!hir.const]
        %21 = hir.recv %arg3[%7] at %17 : !hir.bus<rd i32>[!hir.const] -> i32
        hir.send %8 to %0#0[%8, %7] at %arg10 : !hir.const to tensor<3x!hir.bus<wr i1, wr tuple<i4>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %22 = hir.cast %arg9 : i32 -> i4
        %23 = hir.tuple(%22) : (i4) -> tuple<i4>
        hir.send %23 to %0#0[%8, %8] at %arg10 : tuple<i4> to tensor<3x!hir.bus<wr i1, wr tuple<i4>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %24 = hir.recv %1#1[%8, %7] at %17 : tensor<3x!hir.bus<rd i32>>[!hir.const, !hir.const] -> i32
        hir.send %8 to %0#0[%9, %7] at %arg10 : !hir.const to tensor<3x!hir.bus<wr i1, wr tuple<i4>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %25 = hir.cast %arg9 : i32 -> i4
        %26 = hir.tuple(%25) : (i4) -> tuple<i4>
        hir.send %26 to %0#0[%9, %8] at %arg10 : tuple<i4> to tensor<3x!hir.bus<wr i1, wr tuple<i4>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %27 = hir.recv %1#1[%9, %7] at %17 : tensor<3x!hir.bus<rd i32>>[!hir.const, !hir.const] -> i32
        %28 = hir.delay %arg9 by %8 at %arg10 : i32 -> i32
        hir.send %8 to %2#0[%7, %7] at %17 : !hir.const to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %29 = hir.cast %28 : i32 -> i4
        %30 = hir.tuple(%29, %24) : (i4, i32) -> tuple<i4, i32>
        hir.send %30 to %2#0[%7, %8] at %17 : tuple<i4, i32> to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        hir.send %8 to %2#0[%8, %7] at %17 : !hir.const to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %31 = hir.cast %28 : i32 -> i4
        %32 = hir.tuple(%31, %27) : (i4, i32) -> tuple<i4, i32>
        hir.send %32 to %2#0[%8, %8] at %17 : tuple<i4, i32> to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        hir.send %8 to %2#0[%9, %7] at %17 : !hir.const to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        %33 = hir.cast %28 : i32 -> i4
        %34 = hir.tuple(%33, %21) : (i4, i32) -> tuple<i4, i32>
        hir.send %34 to %2#0[%9, %8] at %17 : tuple<i4, i32> to tensor<3x!hir.bus<wr i1, wr tuple<tuple<i4>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const]
        hir.send %8 to %5#0[%7, %7, %7] at %17 : !hir.const to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
        %35 = hir.tuple(%24) : (i32) -> tuple<i32>
        hir.send %35 to %5#0[%7, %7, %8] at %17 : tuple<i32> to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
        hir.send %8 to %5#0[%8, %7, %7] at %17 : !hir.const to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
        %36 = hir.tuple(%27) : (i32) -> tuple<i32>
        hir.send %36 to %5#0[%8, %7, %8] at %17 : tuple<i32> to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
        hir.send %8 to %5#0[%9, %7, %7] at %17 : !hir.const to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
        %37 = hir.tuple(%21) : (i32) -> tuple<i32>
        hir.send %37 to %5#0[%9, %7, %8] at %17 : tuple<i32> to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
      }
      %16 = hir.delay %15 by %8 at %15 : !hir.time -> !hir.time
      hir.yield at %16
    }
    %13 = hir.for %arg7 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg8 = %arg6 + %8) {
      %15 = hir.for %arg9 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg10 = %arg8 + %8) {
        %17 = hir.delay %arg10 by %8 at %arg10 : !hir.time -> !hir.time
        hir.yield at %17
        %18 = hir.unroll_for %arg11 = 0 to 3 step 1 iter_time( %arg12 = %arg10) {
          hir.yield at %arg12
          %19 = hir.unroll_for %arg13 = 0 to 2 step 1 iter_time( %arg14 = %arg12) {
            hir.yield at %arg14
            %20 = hir.delay %arg14 by %8 at %arg14 : !hir.time -> !hir.time
            hir.send %8 to %3#0[%arg11, %arg13, %7] at %20 : !hir.const to tensor<3x3x!hir.bus<wr i1, wr tuple<>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
            %21 = hir.delay %20 by %8 at %20 : !hir.time -> !hir.time
            %22 = hir.recv %4#1[%arg11, %arg13, %7] at %21 : tensor<3x3x!hir.bus<rd i32>>[!hir.const, !hir.const, !hir.const] -> i32
            %23 = hir.add(%arg13, %8) : (!hir.const, !hir.const) -> (!hir.const)
            hir.send %8 to %5#0[%arg11, %23, %7] at %20 : !hir.const to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
            %24 = hir.tuple(%22) : (i32) -> tuple<i32>
            hir.send %24 to %5#0[%arg11, %23, %8] at %20 : tuple<i32> to tensor<3x3x!hir.bus<wr i1, wr tuple<tuple<>, i32>, proto {join = "join_valid"}>>[!hir.const, !hir.const, !hir.const]
          }
        }
      }
      %16 = hir.delay %15 by %8 at %15 : !hir.time -> !hir.time
      hir.yield at %16
    }
    %14 = hir.for %arg7 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg8 = %arg6 + %8) {
      %15 = hir.for %arg9 : i32 = %7 : !hir.const to %11 : !hir.const step %8 : !hir.const iter_time( %arg10 = %arg8 + %8) {
        %17 = hir.delay %arg10 by %8 at %arg10 : !hir.time -> !hir.time
        hir.yield at %17
        %18 = hir.gt(%arg7, %8) : (i32, !hir.const) -> (i1)
        %19 = hir.gt(%arg9, %8) : (i32, !hir.const) -> (i1)
        %20 = hir.and(%18, %19) : (i1, i1) -> (i1)
        hir.if (%20) at %arg10 {
          %21 = hir.delay %arg10 by %9 at %arg10 : !hir.time -> !hir.time
          %22 = hir.call @weighted_average(%6#0) at %21 : !hir.func<(!hir.memref<3x3xi32, [0, 1], {rd = 1 : i64}>) -> (i32 delay 2)>
          %23 = hir.delay %arg7 by %10 at %arg10 : i32 -> i32
          %24 = hir.delay %arg9 by %10 at %arg10 : i32 -> i32
          %25 = hir.delay %arg10 by %10 at %arg10 : !hir.time -> !hir.time
          hir.send %8 to %arg4[%7] at %25 : !hir.const to !hir.bus<wr i1, wr tuple<tuple<i4, i4>, i32>, proto {join = "join_valid"}>[!hir.const]
          %26 = hir.cast %23 : i32 -> i4
          %27 = hir.cast %24 : i32 -> i4
          %28 = hir.tuple(%26, %27, %22) : (i4, i4, i32) -> tuple<i4, i4, i32>
          hir.send %28 to %arg4[%8] at %25 : tuple<i4, i4, i32> to !hir.bus<wr i1, wr tuple<tuple<i4, i4>, i32>, proto {join = "join_valid"}>[!hir.const]
        }
      }
      %16 = hir.delay %15 by %8 at %15 : !hir.time -> !hir.time
      hir.yield at %16
    }
    hir.return
  }
}

