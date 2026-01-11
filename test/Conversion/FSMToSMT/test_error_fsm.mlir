// RUN: circt-opt -convert-fsm-to-smt %s | FileCheck %s
// CHECK: module {
// CHECK-NEXT:   smt.solver() : () -> () {
// CHECK-NEXT:     %F__0 = smt.declare_fun "F__0" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__1 = smt.declare_fun "F__1" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__2 = smt.declare_fun "F__2" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__3 = smt.declare_fun "F__3" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__4 = smt.declare_fun "F__4" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__5 = smt.declare_fun "F__5" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__6 = smt.declare_fun "F__6" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__7 = smt.declare_fun "F__7" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__8 = smt.declare_fun "F__8" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__9 = smt.declare_fun "F__9" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F__10 = smt.declare_fun "F__10" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %F_ERR = smt.declare_fun "F_ERR" : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:     %0 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<16>):
// CHECK-NEXT:       %c0_bv16 = smt.bv.constant #smt.bv<0> : !smt.bv<16>
// CHECK-NEXT:       %21 = smt.apply_func %F__0(%c0_bv16) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       smt.yield %21 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %0
// CHECK-NEXT:     %1 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__0(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__1(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %1
// CHECK-NEXT:     %2 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__0(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %2
// CHECK-NEXT:     %3 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__1(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__2(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %3
// CHECK-NEXT:     %4 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__1(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %4
// CHECK-NEXT:     %5 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__2(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__3(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %5
// CHECK-NEXT:     %6 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__2(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %6
// CHECK-NEXT:     %7 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__3(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__4(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %7
// CHECK-NEXT:     %8 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__3(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %8
// CHECK-NEXT:     %9 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__4(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__5(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %9
// CHECK-NEXT:     %10 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__4(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %10
// CHECK-NEXT:     %11 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__5(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__6(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %11
// CHECK-NEXT:     %12 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__5(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %12
// CHECK-NEXT:     %13 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__6(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__7(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %13
// CHECK-NEXT:     %14 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__6(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %14
// CHECK-NEXT:     %15 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__7(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__8(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %15
// CHECK-NEXT:     %16 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__7(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %16
// CHECK-NEXT:     %17 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__8(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__9(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %17
// CHECK-NEXT:     %18 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__8(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %18
// CHECK-NEXT:     %19 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__9(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c1_bv16 = smt.bv.constant #smt.bv<1> : !smt.bv<16>
// CHECK-NEXT:       %22 = smt.bv.add %arg2, %c1_bv16 : !smt.bv<16>
// CHECK-NEXT:       %23 = smt.apply_func %F__10(%22) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.distinct %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.ite %24, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.eq %25, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %27 = smt.and %21, %26
// CHECK-NEXT:       %28 = smt.implies %27, %23
// CHECK-NEXT:       smt.yield %28 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %19
// CHECK-NEXT:     %20 = smt.forall {
// CHECK-NEXT:     ^bb0(%arg0: !smt.bv<1>, %arg1: !smt.bv<1>, %arg2: !smt.bv<16>):
// CHECK-NEXT:       %21 = smt.apply_func %F__9(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %22 = smt.apply_func %F_ERR(%arg2) : !smt.func<(!smt.bv<16>) !smt.bool>
// CHECK-NEXT:       %c-1_bv1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %23 = smt.eq %arg1, %c-1_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_0 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %c0_bv1 = smt.bv.constant #smt.bv<0> : !smt.bv<1>
// CHECK-NEXT:       %24 = smt.ite %23, %c-1_bv1_0, %c0_bv1 : !smt.bv<1>
// CHECK-NEXT:       %c-1_bv1_1 = smt.bv.constant #smt.bv<-1> : !smt.bv<1>
// CHECK-NEXT:       %25 = smt.eq %24, %c-1_bv1_1 : !smt.bv<1>
// CHECK-NEXT:       %26 = smt.and %21, %25
// CHECK-NEXT:       %27 = smt.implies %26, %22
// CHECK-NEXT:       smt.yield %27 : !smt.bool
// CHECK-NEXT:     }
// CHECK-NEXT:     smt.assert %20
// CHECK-NEXT:   }
// CHECK-NEXT: }


fsm.machine @fsm10(%err: i1) -> () attributes {initialState = "_0"} {
	%x0 = fsm.variable "x0" {initValue = 0 : i16} : i16
	%c1 = hw.constant 1 : i16
	%c1_i1 = hw.constant 1 : i1
	%c0 = hw.constant 0 : i16


	fsm.state @_0 output {
	} transitions {
		fsm.transition @_1
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_1 output {
	} transitions {
		fsm.transition @_2
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_2 output {
	} transitions {
		fsm.transition @_3
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_3 output {
	} transitions {
		fsm.transition @_4
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_4 output {
	} transitions {
		fsm.transition @_5
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_5 output {
	} transitions {
		fsm.transition @_6
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_6 output {
	} transitions {
		fsm.transition @_7
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_7 output {
	} transitions {
		fsm.transition @_8
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_8 output {
	} transitions {
		fsm.transition @_9
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_9 output {
	} transitions {
		fsm.transition @_10
			guard {
				%tmp1 = comb.icmp ne %err, %c1_i1 : i1
				fsm.return %tmp1
			} action {
				%tmp = comb.add %x0, %c1 : i16
				fsm.update %x0, %tmp : i16
			}
		fsm.transition @ERR
			guard {
				%tmp1 = comb.icmp eq %err, %c1_i1 : i1
				fsm.return %tmp1
			}
	}

	fsm.state @_10 output {
	} transitions {
	}

	fsm.state @ERR output {
	} transitions {
	}
}
