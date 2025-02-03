// RUN: circt-opt %s --arc-find-initial-vectors | FileCheck %s

hw.module @Foo(in %clock: !seq.clock, in %en: i1, in %inA: i3, in %inB: i3) {
  %4 = arc.state @FooMux(%en, %21753, %4) clock %clock latency 1 : (i1, i3, i3) -> i3
  %5 = arc.state @FooMux(%en, %21754, %5) clock %clock latency 1 : (i1, i3, i3) -> i3
  %7 = arc.state @FooMux(%en, %21756, %7) clock %clock latency 1 : (i1, i3, i3) -> i3
  %12 = arc.state @FooMux(%en, %91, %12) clock %clock latency 1 : (i1, i3, i3) -> i3
  %15 = arc.state @FooMux(%en, %93, %15) clock %clock latency 1 : (i1, i3, i3) -> i3
  %16 = arc.state @FooMux(%en, %94, %16) clock %clock latency 1 : (i1, i3, i3) -> i3

  %21753 = comb.xor %200, %inA : i3
  %21754 = comb.xor %201, %inA : i3
  %21756 = comb.xor %202, %inA : i3

  %91 = comb.add %100, %inB : i3
  %93 = comb.add %101, %inB : i3
  %94 = comb.add %102, %inB : i3

  %100 = comb.mul %12, %inA : i3
  %101 = comb.mul %15, %inA : i3
  %102 = comb.sub %16, %inA : i3

  %200 = comb.and %4, %inB : i3
  %201 = comb.and %5, %inB : i3
  %202 = comb.and %7, %inB : i3
}

arc.define @FooMux(%arg0: i1, %arg1: i3, %arg2: i3) -> i3 {
  %0 = comb.mux %arg0, %arg1, %arg2 : i3
  arc.output %0 : i3
}

// CHECK-LABEL:  hw.module @Foo(in %clock : !seq.clock, in %en : i1, in %inA : i3, in %inB : i3) {
// CHECK-NEXT:     [[VEC0:%.+]]:6 = arc.vectorize (%clock, %clock, %clock, %clock, %clock, %clock), (%en, %en, %en, %en, %en, %en), ([[VEC1:%.+]]#0, [[VEC1]]#1, [[VEC1]]#2, [[VEC2:%.+]]#0, [[VEC2]]#1, [[VEC2]]#2), ([[VEC0]]#0, [[VEC0]]#1, [[VEC0]]#2, [[VEC0]]#3, [[VEC0]]#4, [[VEC0]]#5) : (!seq.clock, !seq.clock, !seq.clock, !seq.clock, !seq.clock, !seq.clock, i1, i1, i1, i1, i1, i1, i3, i3, i3, i3, i3, i3, i3, i3, i3, i3, i3, i3) -> (i3, i3, i3, i3, i3, i3) {
// CHECK-NEXT:      ^[[BLOCK:[[:alnum:]]+]](%arg0: !seq.clock, %arg1: i1, %arg2: i3, %arg3: i3):
// CHECK-NEXT:        [[ANS:%.+]] = arc.state @FooMux(%arg1, %arg2, %arg3) clock %arg0 latency 1 : (i1, i3, i3) -> i3
// CHECK-NEXT:        arc.vectorize.return [[ANS:%.+]] : i3
// CHECK-NEXT:      }
// CHECK-NEXT:      [[VEC1]]:3 = arc.vectorize ([[VEC4:%.+]]#0, [[VEC4]]#1, [[VEC4]]#2), (%inA, %inA, %inA) : (i3, i3, i3, i3, i3, i3) -> (i3, i3, i3) {
// CHECK-NEXT:      ^[[BLOCK:[[:alnum:]]+]](%arg0: i3, %arg1: i3):
// CHECK-NEXT:        [[ANS:%.+]] = comb.xor %arg0, %arg1 : i3
// CHECK-NEXT:        arc.vectorize.return [[ANS:%.+]] : i3
// CHECK-NEXT:      }
// CHECK-NEXT:      [[VEC2]]:3 = arc.vectorize ([[VEC3:%.+]]#0, [[VEC3]]#1, [[SCALAR:%.+]]), (%inB, %inB, %inB) : (i3, i3, i3, i3, i3, i3) -> (i3, i3, i3) {
// CHECK-NEXT:      ^[[BLOCK:[[:alnum:]]+]](%arg0: i3, %arg1: i3):
// CHECK-NEXT:       [[ANS:%.+]] = comb.add %arg0, %arg1 : i3
// CHECK-NEXT:       arc.vectorize.return [[ANS:%.+]] : i3
// CHECK-NEXT:      }
// CHECK-NEXT:      [[VEC3]]:2 = arc.vectorize ([[VEC0]]#3, [[VEC0]]#4), (%inA, %inA) : (i3, i3, i3, i3) -> (i3, i3) {
// CHECK-NEXT:      ^[[BLOCK:[[:alnum:]]+]](%arg0: i3, %arg1: i3):
// CHECK-NEXT:        [[ANS:%.+]] = comb.mul %arg0, %arg1 : i3
// CHECK-NEXT:        arc.vectorize.return [[ANS:%.+]] : i3
// CHECK-NEXT:      }
// CHECK-NEXT:      [[SCALAR]] = comb.sub [[VEC0]]#5, %inA : i3
// CHECK-NEXT:      [[VEC4]]:3 = arc.vectorize ([[VEC0]]#0, [[VEC0]]#1, [[VEC0]]#2), (%inB, %inB, %inB) : (i3, i3, i3, i3, i3, i3) -> (i3, i3, i3) {
// CHECK-NEXT:      ^[[BLOCK:[[:alnum:]]+]](%arg0: i3, %arg1: i3):
// CHECK-NEXT:        [[ANS:%.+]] = comb.and %arg0, %arg1 : i3
// CHECK-NEXT:        arc.vectorize.return [[ANS:%.+]] : i3
// CHECK-NEXT:      }
// CHECK-NEXT:      hw.output
// CHECK-NEXT:    }
// CHECK-NEXT:    arc.define @FooMux(%arg0: i1, %arg1: i3, %arg2: i3) -> i3 {
// CHECK-NEXT:      [[ANS:%.+]] = comb.mux %arg0, %arg1, %arg2 : i3
// CHECK-NEXT:      arc.output [[ANS:%.+]] : i3
// CHECK-NEXT:    }
