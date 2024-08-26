func.func @bitvectors() {
  %4 = "smt.forall"() <{weight = 0 : i32}> ({
  ^bb0(%arg0: !smt.int, %arg1: !smt.int):
    %11 = "smt.apply_func"(%0, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %12 = "smt.apply_func"(%1, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %13 = "smt.implies"(%11, %12) : (!smt.bool, !smt.bool) -> !smt.bool
    "smt.yield"(%13) : (!smt.bool) -> ()
  }) : () -> !smt.bool

  %5 = "smt.forall"() <{weight = 0 : i32}> ({
  ^bb0(%arg0: !smt.int, %arg1: !smt.int):
    %12 = "smt.apply_func"(%1, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %13 = "smt.apply_func"(%2, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %14 = "smt.implies"(%12, %13) : (!smt.bool, !smt.bool) -> !smt.bool
    "smt.yield"(%14) : (!smt.bool) -> ()
  }) : () -> !smt.bool

  %6 = "smt.forall"() <{weight = 0 : i32}> ({
  ^bb0(%arg0: !smt.int, %arg1: !smt.int):
    %13 = "smt.apply_func"(%1, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %14 = "smt.apply_func"(%3, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %15 = "smt.implies"(%13, %14) : (!smt.bool, !smt.bool) -> !smt.bool
    "smt.yield"(%15) : (!smt.bool) -> ()
  }) : () -> !smt.bool

  %7 = "smt.forall"() <{weight = 0 : i32}> ({
  ^bb0(%arg0: !smt.int, %arg1: !smt.int):
    %14 = "smt.apply_func"(%2, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %15 = "smt.apply_func"(%1, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %16 = "smt.implies"(%14, %15) : (!smt.bool, !smt.bool) -> !smt.bool
    "smt.yield"(%16) : (!smt.bool) -> ()
  }) : () -> !smt.bool

  %8 = "smt.forall"() <{weight = 0 : i32}> ({
  ^bb0(%arg0: !smt.int, %arg1: !smt.int):
    %15 = "smt.apply_func"(%3, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %16 = "smt.apply_func"(%2, %arg0, %arg1) : (!smt.func<(!smt.int) !smt.bool>, !smt.int, !smt.int) -> !smt.bool
    %17 = "smt.implies"(%15, %16) : (!smt.bool, !smt.bool) -> !smt.bool
    "smt.yield"(%17) : (!smt.bool) -> ()
  }) : () -> !smt.bool
}