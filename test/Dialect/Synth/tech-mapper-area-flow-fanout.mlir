// RUN: circt-opt --pass-pipeline='builtin.module(synth-tech-mapper{strategy=area test=true max-cuts-per-root=8})' %s | FileCheck %s --check-prefixes CHECK,AREA

hw.module @and_inv(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_n(in %a : i1, in %b : i1, out result : i1) attributes {hw.techlib.info = {area = 1.0 : f64, delay = [[1], [1]]}} {
  %0 = synth.aig.and_inv not %a, %b : i1
  hw.output %0 : i1
}

hw.module @and_inv_3_cheap(in %a : i1, in %b : i1, in %c : i1, out result : i1) attributes {hw.techlib.info = {area = 0.75 : f64, delay = [[1], [1], [1]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv not %0, %c : i1
  hw.output %1 : i1
}

hw.module @and3_mid(in %a : i1, in %b : i1, in %c : i1, out result : i1) attributes {hw.techlib.info = {area = 1.75 : f64, delay = [[1], [1], [1]]}} {
  %0 = synth.aig.and_inv %a, %b : i1
  %1 = synth.aig.and_inv %0, %c : i1
  hw.output %1 : i1
}

// Make sure area-flow uses the current cut cover's fanout, not the original
// AIG fanout. The first output switches away from %ab immediately, so the
// second output must see %ab with a single mapped reference.
// CHECK-LABEL: @mapped_fanout_drives_area_flow
hw.module @mapped_fanout_drives_area_flow(in %a : i1, in %b : i1, in %c : i1,
                                          in %d : i1,
                                          out cheap : i1, out recovered : i1) {
  // AREA:      %[[CHEAP:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and_inv_3_cheap(a: %a: i1, b: %b: i1, c: %c: i1) -> (result: i1) {test.arrival_times = [1]}
  // AREA-NEXT: %[[RECOVERED:.+]] = hw.instance "{{[a-zA-Z0-9_]+}}" @and3_mid(a: %a: i1, b: %b: i1, c: %d: i1) -> (result: i1) {test.arrival_times = [1]}
  // AREA-NEXT: hw.output %[[CHEAP]], %[[RECOVERED]] : i1, i1
  %ab = synth.aig.and_inv %a, %b : i1
  %cheap = synth.aig.and_inv %c, not %ab : i1
  %recovered = synth.aig.and_inv %ab, %d : i1
  hw.output %cheap, %recovered : i1, i1
}
