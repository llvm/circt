// RUN: circt-translate --import-verilog %s | FileCheck %s
// RUN: circt-verilog --ir-moore %s
// REQUIRES: slang
// UNSUPPORTED: valgrind

// clang-format off
// CHECK-LABEL: moore.class.classdecl @StringBase {
// CHECK:         moore.class.methoddecl @text : (!moore.class<@StringBase>) -> !moore.string
// CHECK:       }
virtual class StringBase;
  pure virtual function string text();
endclass

// CHECK-LABEL: moore.class.classdecl @StringLeaf extends @StringBase {
// CHECK:         moore.class.methoddecl @text -> @"StringLeaf::text" : (!moore.class<@StringLeaf>) -> !moore.string
// CHECK:       }
// CHECK-LABEL: func.func private @"StringLeaf::text"(%arg0: !moore.class<@StringLeaf>) -> !moore.string {
// CHECK:         return
// CHECK:       }
class StringLeaf extends StringBase;
  virtual function string text();
    return "leaf";
  endfunction
endclass
  // clang-format on
