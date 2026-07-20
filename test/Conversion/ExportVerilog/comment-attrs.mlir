// RUN: circt-opt %s -export-verilog | FileCheck %s

hw.module.extern @Child(in %x : i1)

// CHECK:      // module comment
// CHECK-NEXT: module CommentAttrs
hw.module @CommentAttrs(in %a : i1) attributes {comment = "module comment"} {
  // CHECK:      // wire comment line 1
  // CHECK-NEXT: // wire comment line 2
  // CHECK-NEXT: wire{{ +}}wire_with_comment
  %wire_with_comment = sv.wire {comment = "wire comment line 1\nwire comment line 2"} : !hw.inout<i1>
  sv.assign %wire_with_comment, %a : i1

  // CHECK:      // reg comment
  // CHECK-NEXT: reg{{ +}}reg_with_comment
  %reg_with_comment = sv.reg {comment = "reg comment"} : !hw.inout<i1>
  sv.assign %reg_with_comment, %a : i1

  // CHECK:      // logic comment
  // CHECK-NEXT: logic{{ +}}logic_with_comment
  %logic_with_comment = sv.logic {comment = "logic comment"} : !hw.inout<i1>
  sv.assign %logic_with_comment, %a : i1

  // CHECK:      // instance comment
  // CHECK-NEXT: Child child
  hw.instance "child" @Child(x: %a : i1) -> () {comment = "instance comment"}
}

hw.module @BindParent(in %a : i1) {
  hw.instance "bound_child" sym @bound_child @Child(x: %a : i1) -> () {comment = "bind instance comment", doNotPrint}
}

// CHECK:      // bind instance comment
// CHECK-NEXT: bind BindParent Child bound_child (
sv.bind #hw.innerNameRef<@BindParent::@bound_child>
