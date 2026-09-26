// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file --verify-diagnostics

// Malformed / missing-parameter cases for the FIRRTL debug intrinsics. The
// pre-passes must diagnose these explicitly rather than silently dropping the
// intrinsic and producing incomplete debug metadata. Enum variant data rides
// inline on the var/subfield that uses it, so the enum diagnostics fire from
// `parseInlineEnumVariants` on that intrinsic.
//
// A diagnosed var also leaves its `firrtl.int.generic` unlowered, hence the
// second expected-error on those cases: the converter reports failure so the
// conversion driver fails the pass, rather than letting the tool exit 0 with
// an intrinsic it has already declared broken. Cases that only warn (a name
// that resolves to nothing, unparseable `params`) still lower and drop the
// var, so they carry no such line.

// -----

// Unparseable JSON in inline 'variants'.
firrtl.circuit "EnumBadJSON" {
  firrtl.module @EnumBadJSON(in %s: !firrtl.uint<2>) {
    // expected-error @below {{debug enum: failed to parse 'variants'}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "[{not valid json">
      %s : (!firrtl.uint<2>) -> ()
  }
}

// -----

// Valid JSON that is not an array in 'variants'.
firrtl.circuit "EnumVariantsNotArray" {
  firrtl.module @EnumVariantsNotArray(in %s: !firrtl.uint<2>) {
    // expected-error @below {{debug enum: 'variants' is not a JSON array}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "{}">
      %s : (!firrtl.uint<2>) -> ()
  }
}

// -----

// Variant with a non-integer string value.
firrtl.circuit "EnumBadVariantValue" {
  firrtl.module @EnumBadVariantValue(in %s: !firrtl.uint<2>) {
    // expected-error @below {{debug enum: variant 'A' has non-integer value 'notanint'}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\",\"value\":\"notanint\"}]">
      %s : (!firrtl.uint<2>) -> ()
  }
}

// -----

// Variant missing 'name'.
firrtl.circuit "EnumVariantMissingName" {
  firrtl.module @EnumVariantMissingName(in %s: !firrtl.uint<2>) {
    // expected-error @below {{debug enum: variant is missing 'name'}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "[{\"value\":\"0\"}]">
      %s : (!firrtl.uint<2>) -> ()
  }
}

// -----

// Variant missing 'value'.
firrtl.circuit "EnumVariantMissingValue" {
  firrtl.module @EnumVariantMissingValue(in %s: !firrtl.uint<2>) {
    // expected-error @below {{debug enum: variant 'A' is missing 'value'}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "MyState", enumFqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"A\"}]">
      %s : (!firrtl.uint<2>) -> ()
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

// Wire and memory with the same name 'x': 0-operand debug_var must error
// rather than silently picking the wire.
firrtl.circuit "WireMemSameNameAmbiguity" {
  firrtl.module @WireMemSameNameAmbiguity() {
    %x = firrtl.wire : !firrtl.uint<8>
    %x_mem = chirrtl.combmem {name = "x"} : !chirrtl.cmemory<uint<8>, 4>
    // expected-error @below {{circt_debug_var: name 'x' is ambiguous (matches 2 signals)}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "x", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// Wide enum variant: a value that does not fit `width` is an error (truncating
// it could collapse two variants onto one tag). The diagnostic formats the
// value with APInt::toString, not getZExtValue() (which asserts on > 64 active
// bits and used to abort here).
firrtl.circuit "WideEnumVariant" {
  firrtl.module @WideEnumVariant(in %s: !firrtl.uint<70>) {
    // expected-error @below {{value 75557863725914323419136 does not fit in 70 bits}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "Big", enumFqn: none = "pkg.Big", width: i64 = 70,
       variants: none = "[{\"name\":\"A\",\"value\":\"75557863725914323419136\"}]">
      %s : (!firrtl.uint<70>) -> ()
  }
}

// -----

// Duplicate variant name: a DictionaryAttr would silently collapse the two
// `A` entries, so the converter rejects it up front.
firrtl.circuit "DupVariantName" {
  firrtl.module @DupVariantName(in %s: !firrtl.uint<8>) {
    // expected-error @below {{duplicate variant name 'A'}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "E", enumFqn: none = "pkg.E", width: i64 = 8,
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"},{\"name\":\"A\",\"value\":\"1\"}]">
      %s : (!firrtl.uint<8>) -> ()
  }
}

// -----

// `width` with more than 64 active bits: getZExtValue() would assert, so the
// converter rejects it before reading the value.
firrtl.circuit "EnumWidthTooWide" {
  firrtl.module @EnumWidthTooWide(in %s: !firrtl.uint<8>) {
    // expected-error @below {{'width' parameter exceeds 64 bits}}
    // expected-error @below {{failed to legalize operation 'firrtl.int.generic'}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "E", enumFqn: none = "pkg.E", width: i128 = 18446744073709551616,
       variants: none = "[{\"name\":\"A\",\"value\":\"0\"}]">
      %s : (!firrtl.uint<8>) -> ()
  }
}

// -----

// Malformed `params` JSON on circt_debug_var is non-fatal: the type-parameter
// info is dropped with a warning and the variable is still emitted.
firrtl.circuit "DebugVarBadParams" {
  firrtl.module @DebugVarBadParams(in %x: !firrtl.uint<8>) {
    // expected-warning @below {{debug params JSON failed to parse}}
    firrtl.int.generic "circt_debug_var"
      <name: none = "x", typeName: none = "UInt", params: none = "[not valid json">
      %x : (!firrtl.uint<8>) -> ()
  }
}
