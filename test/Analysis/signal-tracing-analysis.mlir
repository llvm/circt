hw.module @A(%in: i1) -> (out: i1) {
  %0 = hw.constant 1 : i1
  %1 = comb.or %in, %0 : i1
  hw.output %1 : i1
}

hw.module @B(%in: i1) -> (out: i1) {
  %wire = sv.wire : !hw.inout<i1>
  sv.assign %wire, %in : i1
  %0 = sv.read_inout %wire : !hw.inout<i1>
  hw.output %0 : i1
}

hw.module @Top() {
  %input = sv.wire {source} : !hw.inout<i1>
  %0 = hw.constant 1 : i1
  sv.assign %input, %0 : i1
  %1 = sv.read_inout %input : !hw.inout<i1>

  %2 = hw.instance "a" @A(in: %1: i1) -> (out: i1)

  %3 = hw.instance "b" @B(in: %2: i1) -> (out: i1)

  %output = sv.wire {sink} : !hw.inout<i1>
  sv.assign %output, %3 : i1
}
