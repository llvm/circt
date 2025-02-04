// RUN: circt-opt -hw-print-instance-graph %s -o %t 2>&1 | FileCheck %s

// TODO: write the tests

hw.module @Top() {
  hw.instance "alligator" @Alligator() -> ()
  hw.instance "cat" @Cat() -> ()
}

hw.module private @Alligator() {
  hw.instance "bear" @Bear() -> ()
}

hw.module private @Bear() {
  hw.instance "cat" @Cat() -> ()
}

hw.module private @Cat() { }
