// RUN: circt-standalone-opt %s | FileCheck %s

module {
  // CHECK: circt_standalone.foo
  circt_standalone.foo
}
