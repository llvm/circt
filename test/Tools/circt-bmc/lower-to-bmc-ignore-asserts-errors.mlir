// RUN: circt-opt --lower-to-bmc="top-module=testModule bound=10 ignore-asserts-until=11" --split-input-file --verify-diagnostics %s

// expected-error @below {{number of ignored cycles must be less than or equal to bound}}
hw.module @testModule(in %in0 : i1) attributes {num_regs = 0 : i32, initial_values = []} {
  verif.assert %in0 : i1
  hw.output
}
