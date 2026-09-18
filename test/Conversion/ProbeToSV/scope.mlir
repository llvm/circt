// RUN: circt-opt %s --lower-probe-to-sv | FileCheck %s --check-prefix=ROOT
// RUN: circt-opt %s --pass-pipeline='builtin.module(builtin.module(lower-probe-to-sv))' | FileCheck %s --check-prefix=NESTED --implicit-check-not=probe.

module {
  module @nested {
    // ROOT-NOT: hw.hierpath
    // ROOT: hw.module private @Producer
    // ROOT-SAME: out probe : !probe.ref<i8>
    // NESTED: hw.hierpath private @[[PATH:[A-Za-z0-9_]+]] [@Top::@[[INST:[A-Za-z0-9_]+]], @Producer::@[[WIRE:[A-Za-z0-9_]+]]]
    // NESTED: hw.module private @Producer(in %in : i8)
    hw.module private @Producer(in %in: i8,
                                out probe: !probe.ref<i8>) {
      // ROOT: probe.send
      // NESTED: hw.wire %in sym @[[WIRE]] : i8
      %probe = probe.send %in : i8
      hw.output %probe : !probe.ref<i8>
    }

    // ROOT: hw.module @Top
    // NESTED: hw.module @Top
    hw.module @Top(in %in: i8, out out: i8) {
      // ROOT: %[[PROBE:.+]] = hw.instance "producer" @Producer
      // ROOT-SAME: (probe: !probe.ref<i8>)
      // NESTED: hw.instance "producer" sym @[[INST]] @Producer(in: %in: i8) -> ()
      %probe = hw.instance "producer" @Producer(in: %in: i8) ->
          (probe: !probe.ref<i8>)
      // ROOT: probe.read %[[PROBE]]
      // NESTED: sv.xmr.ref @[[PATH]]
      %read = probe.read %probe : <i8>
      hw.output %read : i8
    }
  }
}
