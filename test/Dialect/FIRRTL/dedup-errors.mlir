// RUN: circt-opt --allow-unregistered-dialect -verify-diagnostics -split-input-file -pass-pipeline='firrtl.circuit(firrtl-dedup)' %s

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

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
  firrtl.module @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation is a firrtl.constant}}
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  }
}

// -----

// expected-error@below {{module "Mid1" not deduplicated with "Mid0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Mid0", "~MustDedup|Mid1"]
    }]} {
  firrtl.module @MustDedup() {
    firrtl.instance mid0 @Mid0()
    firrtl.instance mid1 @Mid1()
  }
  firrtl.module @Mid0() {
    // expected-note@below {{first instance targets module "Test0"}}
    firrtl.instance test0 @Test0()
  }
  firrtl.module @Mid1() {
    // expected-note@below {{second instance targets module "Test1"}}
    firrtl.instance test1 @Test1()
  }
  // expected-error@below {{module "Test0" not deduplicated with "Test1"}}
  firrtl.module @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation is a firrtl.constant}}
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operations have different number of results}}
    "test"() : () -> ()
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %0 = "test"() : () -> (i32)
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second type is '!firrtl.uint<2>'}}
    %w = firrtl.wire : !firrtl.uint<2>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result bundle type has different number of elements}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %w = firrtl.wire : !firrtl.bundle<>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{operation result bundle element "a" flip does not match}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second operation here}}
    %w = firrtl.wire : !firrtl.bundle<a flip : uint<1>>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}
// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{bundle element 'a' types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module @Test1() {
    // expected-note@below {{second type is '!firrtl.uint<2>'}}
    %w = firrtl.wire : !firrtl.bundle<b : uint<2>>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of operands}}
    "test"(%a) : (!firrtl.uint<1>) -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() : () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>) {
    // expected-note@below {{operations use different operands, first operand is 'a'}}
    %n = firrtl.node %a : !firrtl.uint<1>
  }
  firrtl.module @Test1(in %c : !firrtl.uint<1>, in %d : !firrtl.uint<1>) {
    // expected-note@below {{second operand is 'd', but should have been 'c'}}
    %n = firrtl.node %d : !firrtl.uint<1>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>, in b : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in c : !firrtl.uint<1>, in d : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of regions}}
    "test"()({}) : () -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() : () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operation regions have different number of blocks}}
    "test"()({
      ^bb0:
        "return"() : () -> ()
    }) : () -> ()
  }
  firrtl.module @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second operation here}}
    "test"() ({
      ^bb0:
        "return"() : () -> ()
      ^bb1:
        "return"() : () -> ()
    }): () -> ()
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{modules have a different number of ports}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second module here}}
  firrtl.module @Test1() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module port 'a' types don't match, first type is '!firrtl.uint<1>'}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second type is '!firrtl.uint<2>'}}
  firrtl.module @Test1(in %a : !firrtl.uint<2>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<2>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{module port 'a' directions don't match, first direction is 'in'}}
  firrtl.module @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second direction is 'out'}}
  firrtl.module @Test1(out %a : !firrtl.uint<1>) { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(out a : !firrtl.uint<1>)
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module @Test0() {
    // expected-note@below {{first block has more operations}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  // expected-note@below {{second block here}}
  firrtl.module @Test1() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{first block here}}
  firrtl.module @Test0() { }
  firrtl.module @Test1() {
    // expected-note@below {{second block has more operations}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{second operation is missing attribute "test1"}}
  firrtl.module @Test0() attributes {test1} { }
  // expected-note@below {{second operation here}}
  firrtl.module @Test1() attributes {} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{first operation has attribute 'test' with value "a"}}
  firrtl.module @Test0() attributes {test = "a"} { }
  // expected-note@below {{second operation has value "b"}}
  firrtl.module @Test1() attributes {test = "b"} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}
