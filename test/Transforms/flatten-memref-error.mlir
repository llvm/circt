// RUN: circt-opt --flatten-memref -verify-diagnostics %s

// Unsupported memref.global initializer kinds should fail conversion.
module {
  // expected-error @+1 {{failed to legalize operation 'memref.global' that was explicitly marked illegal}}
  memref.global "private" constant @g : memref<2x2xi32> = sparse<[[0, 0], [1, 1]], [1, 2]>
}
