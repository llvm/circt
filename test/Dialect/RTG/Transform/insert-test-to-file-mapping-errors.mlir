// RUN: circt-opt --rtg-insert-test-to-file-mapping=split-output=true --verify-diagnostics %s

// expected-error @below {{path must be specified when split-output is set}}
module {
  rtg.test @test0() {}
}
