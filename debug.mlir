hw.module @Basic(in %a: i42, in %b: i42, in %c: i1, out u: i42, out v: i42) {
  %0, %1 = func.call @doSomething(%a, %b) : (i42, i42) -> (i42, i42)
  %2, %3 = scf.if %c -> (i42, i42) {
    %4, %5 = func.call @doSomething(%0, %1) : (i42, i42) -> (i42, i42)
    scf.yield %4, %5 : i42, i42
  } else {
    scf.yield %0, %1 : i42, i42
  }
  hw.output %2, %3 : i42, i42
}

func.func private @doSomething(%arg0: i42, %arg1: i42) -> (i42, i42) {
  %0 = comb.add %arg0, %arg1 : i42
  %1 = comb.mul %arg0, %arg1 : i42
  return %0, %1 : i42, i42
}
