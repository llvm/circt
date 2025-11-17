// RUN: circt-opt --convert-hw-to-smt="for-smtlib-export" --split-input-file --verify-diagnostics %s

// expected-error @below {{multiple hw.module operations are not supported with for-smtlib-export}}
module {
  hw.module @modA(in %in: i32, out out: i32) {
    hw.output %in : i32
  }

  hw.module @modB(in %in: i32, out out: i32) {
    hw.output %in : i32
  }
}

// -----

// expected-error @below {{hw.instance operations are not supported with for-smtlib-export}}
module {
  hw.module.extern @extern()

  hw.module @modC() {
    hw.instance "myExtern" @extern() -> ()
  }
}
