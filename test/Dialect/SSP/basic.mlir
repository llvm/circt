// RUN: circt-opt %s | circt-opt | FileCheck %s

// CHECK: ssp.instance
ssp.instance "Problem" {
}
