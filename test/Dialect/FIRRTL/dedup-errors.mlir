// RUN: circt-opt -verify-diagnostics -split-input-file -pass-pipeline='firrtl.circuit(firrtl-dedup)' %s

// expected-error@below {{MustDeduplicateAnnotation missing "modules" member}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation"
    }]} {
  firrtl.module @MustDedup() { }
}

// -----

// expected-error@below {{MustDeduplicateAnnotation references module "Simple0" which does not exist}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Simple0"]
    }]} {
  firrtl.module @MustDedup() { }
}

// -----

// expected-error@below {{module "~MustDedup|Test1" not deduplicated with "~MustDedup|Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %i : !firrtl.uint<1>) { }
  firrtl.module @Test1(in %i : !firrtl.uint<8>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in i : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in i : !firrtl.uint<8>)
  }
}
