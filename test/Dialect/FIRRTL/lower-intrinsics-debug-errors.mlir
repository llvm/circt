// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file --verify-diagnostics

// Malformed / missing-parameter cases for the Chisel debug intrinsics. The
// pre-passes must diagnose these explicitly rather than silently dropping the
// intrinsic and producing incomplete debug metadata.

// -----

// Missing 'fqn' param on circt_debug_enumdef.
firrtl.circuit "EnumDefMissingFqn" {
  firrtl.module @EnumDefMissingFqn() {
    // expected-error @below {{circt_debug_enumdef: missing required parameter}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState",
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"}]"> : () -> ()
  }
}

// -----

// Missing 'typeName' param on circt_debug_enumdef.
firrtl.circuit "EnumDefMissingTypeName" {
  firrtl.module @EnumDefMissingTypeName() {
    // expected-error @below {{circt_debug_enumdef: missing required parameter}}
    firrtl.int.generic "circt_debug_enumdef"
      <fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"}]"> : () -> ()
  }
}

// -----

// Missing 'variants' param on circt_debug_enumdef.
firrtl.circuit "EnumDefMissingVariants" {
  firrtl.module @EnumDefMissingVariants() {
    // expected-error @below {{circt_debug_enumdef: missing required parameter}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$"> : () -> ()
  }
}

// -----

// Unparseable JSON in 'variants'.
firrtl.circuit "EnumDefBadJSON" {
  firrtl.module @EnumDefBadJSON() {
    // expected-error @below {{circt_debug_enumdef: failed to parse 'variants'}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{not valid json"> : () -> ()
  }
}

// -----

// Valid JSON that is not an array in 'variants'. Previously this silently
// produced a dbg.enumdef with an empty variant map.
firrtl.circuit "EnumDefVariantsNotArray" {
  firrtl.module @EnumDefVariantsNotArray() {
    // expected-error @below {{circt_debug_enumdef: 'variants' is not a JSON array}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "{}"> : () -> ()
  }
}

// -----

// Variant with a non-integer string value. Previously this silently produced a
// variant with value 0.
firrtl.circuit "EnumDefBadVariantValue" {
  firrtl.module @EnumDefBadVariantValue() {
    // expected-error @below {{circt_debug_enumdef: variant 'A' has non-integer value 'notanint'}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\",\"value\":\"notanint\"}]"> : () -> ()
  }
}

// -----

// Variant missing 'name'.
firrtl.circuit "EnumDefVariantMissingName" {
  firrtl.module @EnumDefVariantMissingName() {
    // expected-error @below {{circt_debug_enumdef: variant is missing 'name'}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"value\":\"0\"}]"> : () -> ()
  }
}

// -----

// Variant missing 'value'.
firrtl.circuit "EnumDefVariantMissingValue" {
  firrtl.module @EnumDefVariantMissingValue() {
    // expected-error @below {{circt_debug_enumdef: variant 'A' is missing 'value'}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\"}]"> : () -> ()
  }
}

// -----

// Two enumdefs with the same fqn but differing variants; the second must
// not create a new dbg.enumdef, but the mismatch must be warned about.
firrtl.circuit "EnumDefConflict" {
  firrtl.module @EnumDefConflict() {
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"}]"> : () -> ()
    // expected-warning @below {{duplicate circt_debug_enumdef for fqn 'pkg.MyState$' with differing variants}}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"},{\"name\":\"B\",\"value\":\"1\"}]"> : () -> ()
  }
}

// -----

// Missing 'name' on circt_debug_subfield.
firrtl.circuit "SubfieldMissingName" {
  firrtl.module @SubfieldMissingName() {
    %io = firrtl.wire : !firrtl.bundle<x: uint<8>>
    // expected-error @below {{circt_debug_subfield: missing required parameter 'name'}}
    firrtl.int.generic "circt_debug_subfield"
      <typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()
  }
}

// -----

// Missing 'parent' on circt_debug_subfield; without it the pre-pass cannot
// link the leaf to its circt_debug_var.
firrtl.circuit "SubfieldMissingParent" {
  firrtl.module @SubfieldMissingParent() {
    %io = firrtl.wire : !firrtl.bundle<x: uint<8>>
    // expected-error @below {{circt_debug_subfield: missing required parameter 'parent'}}
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.x", typeName: none = "UInt">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()
  }
}

// -----

// 0-operand circt_debug_var whose 'name' matches no wire/port/reg: converter
// emits a warning and erases the op (see FIRRTLIntrinsics.cpp,
// CirctDebugVarConverter::convert, the `!rawSignal` branch).
firrtl.circuit "DebugVarUnresolved" {
  firrtl.module @DebugVarUnresolved() {
    // expected-warning @below {{circt_debug_var: no wire, port, or register named 'missing' found}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "missing", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// Two circt_debug_var with the same name="x" in one module; the second must
// be diagnosed so metadata consumers (hgdb/tywaves) are not silently given a
// duplicate variable entry. Implementation: walk of existing `dbg.variable`
// ops at convert time (FIRRTLIntrinsics.cpp CirctDebugVarConverter::convert).
firrtl.circuit "DebugVarDuplicateName" {
  firrtl.module @DebugVarDuplicateName(in %x: !firrtl.uint<8>,
                                       in %y: !firrtl.uint<8>) {
    firrtl.int.generic "circt_debug_var"
      <name: none = "x", typeName: none = "UInt">
      %x : (!firrtl.uint<8>) -> ()
    // expected-warning @below {{duplicate circt_debug_var with name 'x'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "x", typeName: none = "UInt">
      %y : (!firrtl.uint<8>) -> ()
  }
}

// -----

// 'name' is not rooted at 'parent'; frontend regression check.
firrtl.circuit "SubfieldNameNotRooted" {
  firrtl.module @SubfieldNameNotRooted() {
    %io = firrtl.wire : !firrtl.bundle<x: uint<8>>
    // expected-error @below {{circt_debug_subfield: 'name' (state) is not rooted at 'parent' (io)}}
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "state", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()
  }
}

// -----

// Empty 'name' on circt_debug_subfield (finding #5).
firrtl.circuit "SubfieldEmptyName" {
  firrtl.module @SubfieldEmptyName() {
    %io = firrtl.wire : !firrtl.bundle<x: uint<8>>
    // expected-error @below {{circt_debug_subfield: 'name' must not be empty}}
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()
  }
}

// -----

// Empty 'parent' on circt_debug_subfield (finding #5).
firrtl.circuit "SubfieldEmptyParent" {
  firrtl.module @SubfieldEmptyParent() {
    %io = firrtl.wire : !firrtl.bundle<x: uint<8>>
    // expected-error @below {{circt_debug_subfield: 'parent' must not be empty}}
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.x", typeName: none = "UInt", parent: none = "">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()
  }
}

// -----

// Dangling enumFqn: circt_debug_var references an enumFqn for which no
// circt_debug_enumdef exists. A warning must be emitted but UHDI lowering
// continues (finding #4).
firrtl.circuit "DanglingEnumFqn" {
  firrtl.module @DanglingEnumFqn(in %s: !firrtl.uint<2>) {
    // expected-warning @below {{no circt_debug_enumdef found for 'pkg.Missing$'; leaf will be emitted without enum binding}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "UInt", enumFqn: none = "pkg.Missing$">
      %s : (!firrtl.uint<2>) -> ()
  }
}
