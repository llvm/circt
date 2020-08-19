// RUN: circt-opt %s | FileCheck %s

// CHECK: func
func @empty_loop() -> () {
  affine.for %i = 0 to 10 {
    affine.yield
  }

  return
}