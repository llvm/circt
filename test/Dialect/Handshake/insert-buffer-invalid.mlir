// RUN: circt-opt %s -handshake-insert-buffers=strategy=foo -verify-diagnostics

module {
  // expected-error @+1 {{Unknown buffer strategy: foo}}
  handshake.func @test(%arg0: none, ...) -> none {
    return %arg0 : none
  }
}
