// RUN: circt-opt --allow-unregistered-dialect -verify-diagnostics -split-input-file -pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s

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
// expected-error@below {{module "Test3" not deduplicated with "Test2"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    },
    {
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test2", "~MustDedup|Test3"]
    }]} {
  // expected-note@below {{module marked NoDedup}}
  firrtl.module private @Test0() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  firrtl.module private @Test1() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  // expected-note@below {{module marked NoDedup}}
  firrtl.module private @Test2() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  firrtl.module private @Test3() { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
    firrtl.instance test2 @Test2()
    firrtl.instance test3 @Test3()
  }
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
  firrtl.module private @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module private @Test1() {
    // expected-note@below {{second operation is a firrtl.constant}}
    %c1_ui8 = firrtl.constant 1 : !firrtl.uint<8>
  }
}

// -----

// expected-error@+2 {{module "Mid1" not deduplicated with "Mid0"}}
// expected-note@+1 {{in instance "test0" of "Test0", and instance "test1" of "Test1"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Mid0", "~MustDedup|Mid1"]
    }]} {
  firrtl.module @MustDedup() {
    firrtl.instance mid0 @Mid0()
    firrtl.instance mid1 @Mid1()
  }
  firrtl.module private @Mid0() {
    firrtl.instance test0 @Test0()
  }
  firrtl.module private @Mid1() {
    firrtl.instance test1 @Test1()
  }
  firrtl.module private @Test0() {
    // expected-note@below {{first operation is a firrtl.wire}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() {
    // expected-note@below {{operations have different number of results}}
    "test"() : () -> ()
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() {
    // expected-note@below {{operation result types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.uint<1>
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() {
    // expected-note@below {{operation result bundle type has different number of elements}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() {
    // expected-note@below {{operation result bundle element "a" flip does not match}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() {
    // expected-note@below {{bundle element 'a' types don't match, first type is '!firrtl.uint<1>'}}
    %w = firrtl.wire : !firrtl.bundle<a : uint<1>>
  }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of operands}}
    "test"(%a) : (!firrtl.uint<1>) -> ()
  }
  firrtl.module private @Test1(in %a : !firrtl.uint<1>) {
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>, in %b : !firrtl.uint<1>) {
    // expected-note@below {{operations use different operands, first operand is 'a'}}
    %n = firrtl.node %a : !firrtl.uint<1>
  }
  firrtl.module private @Test1(in %c : !firrtl.uint<1>, in %d : !firrtl.uint<1>) {
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operations have different number of regions}}
    "test"()({}) : () -> ()
  }
  firrtl.module private @Test1(in %a : !firrtl.uint<1>) {
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{operation regions have different number of blocks}}
    "test"()({
      ^bb0:
        "return"() : () -> ()
    }) : () -> ()
  }
  firrtl.module private @Test1(in %a : !firrtl.uint<1>) {
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

// Check same number of blocks but instructions across are same.
// https://github.com/llvm/circt/issues/7415
// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "SameInstDiffBlock" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~SameInstDiffBlock|Test0", "~SameInstDiffBlock|Test1"]
    }]} {
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) {
    "test"()({
      ^bb0:
        // expected-note@below {{first block has more operations}}
        "return"() : () -> ()
    }, {
      ^bb0:
    }) : () -> ()
  }
  firrtl.module private @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second block here}}
    "test"() ({
      ^bb0:
    }, {
      ^bb0:
        "return"() : () -> ()
    }): () -> ()
  }
  firrtl.module @SameInstDiffBlock() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// Check differences in block arguments.
// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "BlockArgTypes" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~BlockArgTypes|Test0", "~BlockArgTypes|Test1"]
    }]} {
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) {
    // expected-note@below {{types don't match, first type is 'i32'}}
    "test"()({
      ^bb0(%arg0 : i32):
        "return"() : () -> ()
    }) : () -> ()
  }
  firrtl.module private @Test1(in %a : !firrtl.uint<1>) {
    // expected-note@below {{second type is 'i64'}}
    "test"() ({
      ^bb0(%arg0 : i64):
        "return"() : () -> ()
    }): () -> ()
  }
  firrtl.module @BlockArgTypes() {
    firrtl.instance test0 @Test0(in a : !firrtl.uint<1>)
    firrtl.instance test1 @Test1(in a : !firrtl.uint<1>)
  }
}

// -----

// Check empty block not same as no block.
// https://github.com/llvm/circt/issues/7416
// expected-error@below {{module "B" not deduplicated with "A"}}
firrtl.circuit "NoBlockEmptyBlock" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~NoBlockEmptyBlock|A", "~NoBlockEmptyBlock|B"]
    }]} {
  firrtl.module private @A(in %x: !firrtl.uint<1>) {
    // expected-note @below {{operation regions have different number of blocks}}
    firrtl.when %x : !firrtl.uint<1> {
    }
  }
  firrtl.module private @B(in %x: !firrtl.uint<1>) {
    // expected-note @below {{second operation here}}
    firrtl.when %x : !firrtl.uint<1> {
    } else {
    }
  }
  firrtl.module @NoBlockEmptyBlock(in %x: !firrtl.uint<1>) {
    %a_x = firrtl.instance a @A(in x: !firrtl.uint<1>)
    firrtl.matchingconnect %a_x, %x : !firrtl.uint<1>
    %b_x = firrtl.instance b @B(in x: !firrtl.uint<1>)
    firrtl.matchingconnect %b_x, %x : !firrtl.uint<1>
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  // expected-note@below {{port 'a' only exists in one of the modules}}
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second module to be deduped that does not have the port}}
  firrtl.module private @Test1() { }
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second type is '!firrtl.uint<2>'}}
  firrtl.module private @Test1(in %a : !firrtl.uint<2>) { }
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
  firrtl.module private @Test0(in %a : !firrtl.uint<1>) { }
  // expected-note@below {{second direction is 'out'}}
  firrtl.module private @Test1(out %a : !firrtl.uint<1>) { }
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
  firrtl.module private @Test0() {
    // expected-note@below {{first block has more operations}}
    %w = firrtl.wire : !firrtl.uint<8>
  }
  // expected-note@below {{second block here}}
  firrtl.module private @Test1() { }
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
  firrtl.module private @Test0() { }
  firrtl.module private @Test1() {
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
  firrtl.module private @Test0() attributes {test1} { }
  // expected-note@below {{second operation here}}
  firrtl.module private @Test1() attributes {} { }
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
  firrtl.module private @Test0() attributes {test = "a"} { }
  // expected-note@below {{second operation has value "b"}}
  firrtl.module private @Test1() attributes {test = "b"} { }
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
  // expected-note@below {{first operation has attribute 'test' with value 0x21}}
  firrtl.module private @Test0() attributes {test = 33 : i8} { }
  // expected-note@below {{second operation has value 0x20}}
  firrtl.module private @Test1() attributes {test = 32 : i8} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
  }
}

// -----

// This test is checking that we don't crash when the two modules we want
// deduped were actually deduped with another module.

// expected-error@below {{module "Test3" not deduplicated with "Test1"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test1", "~MustDedup|Test3"]
    }]} {

  // expected-note@below {{first operation has attribute 'test' with value "a"}}
  firrtl.module private @Test0() attributes {test = "a"} { }
  firrtl.module private @Test1() attributes {test = "a"} { }
  // expected-note@below {{second operation has value "b"}}
  firrtl.module private @Test2() attributes {test = "b"} { }
  firrtl.module private @Test3() attributes {test = "b"} { }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0()
    firrtl.instance test1 @Test1()
    firrtl.instance test2 @Test2()
    firrtl.instance test3 @Test3()
  }
}


// -----

// expected-error@below {{module "Bar" not deduplicated with "Foo"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Foo", "~MustDedup|Bar"]
    }]} {

  // expected-note@below {{module is in dedup group 'foo'}}
  firrtl.module private @Foo() attributes {annotations = [{
    class = "firrtl.transforms.DedupGroupAnnotation",
    group = "foo"
  }]} { }

  // expected-note@below {{module is not part of a dedup group}}
  firrtl.module private @Bar() { }

  firrtl.module @MustDedup() {
    firrtl.instance foo @Foo()
    firrtl.instance bar  @Bar()
  }
}

// -----

firrtl.circuit "MustDedup" attributes {} {

  // expected-error@below {{module belongs to multiple dedup groups: "foo", "bar"}}
  firrtl.module private @Child() attributes {annotations = [
    {
      class = "firrtl.transforms.DedupGroupAnnotation",
      group = "foo"
    },
    {
      class = "firrtl.transforms.DedupGroupAnnotation",
      group = "bar"
    }
  ]} { }

  firrtl.module @MustDedup() {
    firrtl.instance c @Child()
  }
}

// -----

// expected-error@below {{module "Test1" not deduplicated with "Test0"}}
firrtl.circuit "MustDedup" attributes {annotations = [{
      class = "firrtl.transforms.MustDeduplicateAnnotation",
      modules = ["~MustDedup|Test0", "~MustDedup|Test1"]
    }]} {
  firrtl.module private @Test0() {
    %w0 = firrtl.wire sym [<@sym, 1, private>]: !firrtl.vector<uint<1>, 2>
    // expected-note @below {{operations have different targets, first operation has field 1 of op %w0 = firrtl.wire sym [<@sym,1,private>] : !firrtl.vector<uint<1>, 2>}}
    %1 = firrtl.ref.rwprobe <@Test0::@sym> : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module private @Test1() {
    %w1 = firrtl.wire sym [<@sym, 2, private>]: !firrtl.vector<uint<1>, 2>
    // expected-note @below {{second operation has field 2 of op %w1 = firrtl.wire sym [<@sym,2,private>] : !firrtl.vector<uint<1>, 2>}}
    %0 = firrtl.ref.rwprobe <@Test1::@sym> : !firrtl.rwprobe<uint<1>>
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
  firrtl.module private @Test0(in %in : !firrtl.vector<uint<1>, 2> sym [<@sym, 1, private>]) {
    // expected-note @below {{operations have different targets, first operation has field 1 of port 0 on @Test0}}
    %0 = firrtl.ref.rwprobe <@Test0::@sym> : !firrtl.rwprobe<uint<1>>
  }
  firrtl.module private @Test1(in %in : !firrtl.vector<uint<1>, 2> sym [<@sym, 2, private>]) {
    // expected-note @below {{second operation has field 2 of port 0 on @Test1}}
    %0 = firrtl.ref.rwprobe <@Test1::@sym>: !firrtl.rwprobe<uint<1>>
  }
  firrtl.module @MustDedup() {
    firrtl.instance test0 @Test0(in in : !firrtl.vector<uint<1>, 2>)
    firrtl.instance test1 @Test1(in in : !firrtl.vector<uint<1>, 2>)
  }
}
