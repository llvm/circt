// RUN: circt-opt -feedthrough='source-module=A source-ports=a,b inner-modules=B,C dest-module=D' | FileCheck %s

hw.module private @A(%a: i1, %b: i1, %c: i1) -> (d: i1) {
  hw.output %c : i1
}

hw.module private @B(%e: i1) -> (f: i1) {
  hw.output %e : i1
}

hw.module private @C() -> () {
  hw.output
}

hw.module private @D(%g: i1) {
}

hw.module @Top(%h: i1, %i: i1, %j: i1) -> () {
  %a0.d = hw.instance "a0" @A(a: %h: i1, b: %i: i1, c: %j: i1) -> (d: i1)
  %b0.f = hw.instance "b0" @B(e: %a0.d: i1) -> (f: i1)
  hw.instance "c0" @C() -> ()
  hw.instance "d0" @D(g: %b0.f: i1) -> ()
}
