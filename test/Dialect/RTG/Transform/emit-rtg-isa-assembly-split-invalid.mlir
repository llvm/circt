// RUN: circt-opt --rtg-emit-isa-assembly=split-output=true --verify-diagnostics %s

// expected-error @below {{'split-output' option only valid in combination with a valid 'path' argument}}
module {
  rtg.test @test0() {}
}
