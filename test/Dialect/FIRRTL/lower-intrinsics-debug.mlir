// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file | FileCheck %s

// Scalar, bundle, and vec cases for the three FIRRTL debug intrinsics
// (circt_debug_moduleinfo, circt_debug_var, circt_debug_subfield) and
// circt_debug_enumdef.

// moduleinfo lowers to a discardable attribute on the module (empty params
// array is forwarded as []); DictionaryAttr keys sort lexicographically.
// CHECK-LABEL: firrtl.module @DebugTest
// CHECK-SAME: attributes {dbg.moduleinfo = {params = [], typeName = "MyModule"}}
firrtl.circuit "DebugTest" {
  firrtl.module @DebugTest(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {

    // 1. moduleinfo -> module attribute, intrinsic erased
    firrtl.int.generic "circt_debug_moduleinfo"
      <typeName: none = "MyModule", params: none = "[]"> : () -> ()

    // 2. enumdef with JSON array (value serialized as string by frontend).
    //    No module-level op is materialised; the variant data is staged and
    //    realised as an inline `dbg.enum` at each use site.
    // CHECK-NOT: dbg.enumdef
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    // 3. circt_debug_var on scalar wire -> dbg.value + dbg.variable
    // CHECK: %[[W:.+]] = dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK: dbg.variable "w", %[[W]] : !dbg.value
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w", typeName: none = "UInt">
      %w : (!firrtl.uint<8>) -> ()

    // 4. circt_debug_var on a passive bundle -> dbg.struct + dbg.variable
    //    The converter builds:
    //      %a = firrtl.subfield %io[in]
    //      %b = firrtl.subfield %io[out]
    //      %s = dbg.struct {"in": %a, "out": %b}
    //      %x = dbg.value %s typeName "MyBundle"
    //      dbg.variable "io", %x
    // CHECK:      firrtl.subfield %io[in]
    // CHECK:      firrtl.subfield %io[out]
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK: %[[IO:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK: dbg.variable "io", %[[IO]] : !dbg.value
    %io = firrtl.wire : !firrtl.bundle<in: uint<8>, out: uint<8>>
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<in: uint<8>, out: uint<8>>) -> ()

    // 5. circt_debug_var on a Vec -> dbg.array + dbg.variable
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %v[1]
    // CHECK:      %[[A:.+]] = dbg.array
    // CHECK-NEXT: %[[V:.+]] = dbg.value %[[A]] typeName "Vec" : !dbg.array
    // CHECK-NEXT: dbg.variable "v", %[[V]] : !dbg.value
    %v = firrtl.wire : !firrtl.vector<uint<4>, 2>
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "Vec">
      %v : (!firrtl.vector<uint<4>, 2>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with an enum-typed field -- the field must get a dbg.value
// wrapping it with an enumDef reference.
//
// A frontend emits:
//   circt_debug_enumdef for MyState
//   circt_debug_subfield(parent="io", name="state") for io.state with enumFqn
//   circt_debug_subfield(parent="io", name="data")  for io.data
//   circt_debug_var(name="io")                      for io

// CHECK-LABEL: firrtl.module @BundleEnumFieldTest
firrtl.circuit "BundleEnumFieldTest" {
  firrtl.module @BundleEnumFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    // No module-level enumdef op is materialised; the variant data is staged.
    // CHECK-NOT: dbg.enumdef
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    %io = firrtl.wire : !firrtl.bundle<state: uint<2>, data: uint<8>>

    // Leaf io.state with enumFqn -- captured into leafMetaMap, erased
    // CHECK-NOT: circt_debug_subfield
    // CHECK-NOT: dbg.variable "state"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.state", typeName: none = "UInt", parent: none = "io",
       enumTypeName: none = "MyState", enumFqn: none = "pkg.MyState$">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Leaf io.data without enum
    // CHECK-NOT: dbg.variable "data"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.data", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    // Root circt_debug_var -- produces dbg.struct + dbg.variable. The enum
    // leaf's value is wrapped in an inline `dbg.enum` carrying the variants.
    // CHECK:      firrtl.subfield %{{.*}}[state]
    // CHECK:      %[[E:.+]] = dbg.enum %{{.*}}, "MyState", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.MyState$" : !firrtl.uint<2>
    // CHECK:      dbg.value %[[E]] typeName "UInt" : !dbg.enum
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK:      %[[IO:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK:      dbg.variable "io", %[[IO]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with a true FEnumType field -- FEnumType::isGround() returns false,
// so the field must be routed through the FEnumType case in build(), not
// dropped by the FIRRTLBaseType fallback.

// CHECK-LABEL: firrtl.module @BundleFEnumFieldTest
firrtl.circuit "BundleFEnumFieldTest" {
  firrtl.module @BundleFEnumFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    // CHECK-NOT: dbg.enumdef
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    %io = firrtl.wire : !firrtl.bundle<state: enum<Idle: uint<0>, Run: uint<0>>, data: uint<8>>

    // CHECK-NOT: circt_debug_subfield
    // CHECK-NOT: dbg.variable "state"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.state", typeName: none = "UInt", parent: none = "io",
       enumTypeName: none = "MyState", enumFqn: none = "pkg.MyState$">
      %io : (!firrtl.bundle<state: enum<Idle: uint<0>, Run: uint<0>>, data: uint<8>>) -> ()

    // CHECK-NOT: dbg.variable "data"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.data", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<state: enum<Idle: uint<0>, Run: uint<0>>, data: uint<8>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[state]
    // CHECK:      %[[E:.+]] = dbg.enum %{{.*}}, "MyState", {Idle = 0 : i64, Run = 1 : i64} fqn "pkg.MyState$" : !firrtl.enum<Idle, Run>
    // CHECK:      dbg.value %[[E]] typeName "UInt" : !dbg.enum
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK:      %[[IO:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK:      dbg.variable "io", %[[IO]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<state: enum<Idle: uint<0>, Run: uint<0>>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with a FixedPoint field carrying type parameters.
// Verifies that the "params" JSON string is parsed and forwarded to
// dbg.value as an ArrayAttr of DictionaryAttrs.

// CHECK-LABEL: firrtl.module @BundleParamsFieldTest
firrtl.circuit "BundleParamsFieldTest" {
  firrtl.module @BundleParamsFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    %io = firrtl.wire : !firrtl.bundle<fp: uint<8>, data: uint<8>>

    // Leaf io.fp with params
    // CHECK-NOT: dbg.variable "fp"
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.fp", typeName: none = "FixedPoint", parent: none = "io",
       params: none = "[{\"name\":\"width\",\"value\":\"8\"},{\"name\":\"binaryPoint\",\"value\":\"4\"}]">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    // Leaf io.data without params
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.data", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    // Root circt_debug_var -- produces dbg.struct + dbg.variable
    // CHECK:      firrtl.subfield %{{.*}}[fp]
    // CHECK:      dbg.value %{{.*}} typeName "FixedPoint" params [{name = "width", value = "8"}, {name = "binaryPoint", value = "4"}] : !firrtl.uint<8>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK:      %[[IO:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK:      dbg.variable "io", %[[IO]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// moduleinfo with constructor params

// CHECK-LABEL: firrtl.module @ModuleInfoWithParams
// CHECK-SAME: attributes {dbg.moduleinfo = {params = [{name = "width", value = "8"}], typeName = "Counter"}}
firrtl.circuit "ModuleInfoWithParams" {
  firrtl.module @ModuleInfoWithParams() {
    firrtl.int.generic "circt_debug_moduleinfo"
      <typeName: none = "Counter",
       params: none = "[{\"name\":\"width\",\"value\":\"8\"}]">
      : () -> ()
  }
}

// -----

// 0-operand circt_debug_var: the frontend emits no SSA operand when the
// referenced value is a non-passive aggregate (e.g. a bidirectional bundle
// port) whose SSA form is not directly assignable to a dbg.variable. The
// converter must locate the matching port / wire / node / reg by name.

// CHECK-LABEL: firrtl.module @ZeroOperandVarPort
firrtl.circuit "ZeroOperandVarPort" {
  firrtl.module @ZeroOperandVarPort(
      in %a: !firrtl.uint<8>,
      out %b: !firrtl.uint<8>) {
    // Match by port name; resolves to the block argument %a.
    // CHECK: %[[V:.+]] = dbg.value %a typeName "UInt" : !firrtl.uint<8>
    // CHECK: dbg.variable "a", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "a", typeName: none = "UInt"> : () -> ()

    firrtl.connect %b, %a : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// 0-operand circt_debug_var resolving to a wire.

// CHECK-LABEL: firrtl.module @ZeroOperandVarWire
firrtl.circuit "ZeroOperandVarWire" {
  firrtl.module @ZeroOperandVarWire() {
    // CHECK: %mywire = firrtl.wire
    %mywire = firrtl.wire : !firrtl.uint<8>
    // CHECK: %[[V:.+]] = dbg.value %mywire typeName "UInt" : !firrtl.uint<8>
    // CHECK: dbg.variable "mywire", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "mywire", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// 0-operand circt_debug_var with no matching port/wire/reg is silently erased
// (a warning is emitted by the converter; the warning case is asserted by
// `DebugVarUnresolved` in lower-intrinsics-debug-errors.mlir). Here we just
// confirm no `dbg.variable` is produced.

// CHECK-LABEL: firrtl.module @ZeroOperandVarNoMatch
firrtl.circuit "ZeroOperandVarNoMatch" {
  firrtl.module @ZeroOperandVarNoMatch() {
    // CHECK-NOT: dbg.variable
    // CHECK-NOT: circt_debug_var
    firrtl.int.generic "circt_debug_var"
      <name: none = "nonexistent", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// 0-operand circt_debug_var resolving to a firrtl.reg.

// CHECK-LABEL: firrtl.module @ZeroOperandVarReg
firrtl.circuit "ZeroOperandVarReg" {
  firrtl.module @ZeroOperandVarReg(in %clk: !firrtl.clock) {
    // CHECK: %myreg = firrtl.reg
    %myreg = firrtl.reg %clk : !firrtl.clock, !firrtl.uint<8>
    // CHECK: %[[V:.+]] = dbg.value %myreg typeName "UInt" : !firrtl.uint<8>
    // CHECK: dbg.variable "myreg", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "myreg", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// 0-operand circt_debug_var resolving to a firrtl.regreset (async reset)
// with a bundle type.  The fallback name-walk must handle RegResetOp the same
// way it handles RegOp.

// CHECK-LABEL: firrtl.module @ZeroOperandVarRegReset
firrtl.circuit "ZeroOperandVarRegReset" {
  firrtl.module @ZeroOperandVarRegReset(
      in %clk: !firrtl.clock,
      in %rst: !firrtl.asyncreset) {
    // CHECK: %r = firrtl.regreset
    %c0 = firrtl.aggregateconstant [0 : ui8, 0 : ui8] : !firrtl.bundle<x: uint<8>, y: uint<8>>
    %r = firrtl.regreset %clk, %rst, %c0 :
        !firrtl.clock, !firrtl.asyncreset,
        !firrtl.bundle<x: uint<8>, y: uint<8>>,
        !firrtl.bundle<x: uint<8>, y: uint<8>>
    // CHECK:      firrtl.subfield %r[x]
    // CHECK:      firrtl.subfield %r[y]
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK:      %[[R:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK:      dbg.variable "r", %[[R]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "r", typeName: none = "MyBundle"> : () -> ()
  }
}

// -----

// Nested aggregate: bundle whose field is itself a bundle. Exercises
// recursion in DebugAggregateBuilder: the outer struct contains an inner
// struct, with a leaf-meta entry on the deepest field.

// CHECK-LABEL: firrtl.module @NestedBundleTest
firrtl.circuit "NestedBundleTest" {
  firrtl.module @NestedBundleTest() {
    %io = firrtl.wire : !firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>

    // Subfield meta on a deeply nested leaf: io.inner.a
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.inner.a", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[inner]
    // CHECK:      firrtl.subfield %{{.*}}[a]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<4>
    // CHECK:      firrtl.subfield %{{.*}}[b]
    // CHECK:      dbg.struct
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "Outer">
      %io : (!firrtl.bundle<inner: bundle<a: uint<4>, b: uint<4>>>) -> ()
  }
}

// -----

// circt_debug_typedef is silently consumed: no representation in the IR yet,
// and no error.

// CHECK-LABEL: firrtl.module @TypeDefDropped
firrtl.circuit "TypeDefDropped" {
  firrtl.module @TypeDefDropped() {
    // CHECK-NOT: circt_debug_typedef
    firrtl.int.generic "circt_debug_typedef"
      <typeName: none = "MyAlias"> : () -> ()
  }
}

// -----

// Nested bundle: bundle<a: bundle<b: uint<8>>>.
// A single circt_debug_subfield for the leaf "io.a.b" (parent="io") and a
// circt_debug_var for the root "io".  The converter must recurse two levels:
//   outer struct {"a": inner_struct}
//   inner struct {"b": dbg.value with the leaf meta}
//
// Expected lowering:
//   firrtl.subfield %io[a]            -- descend into outer field
//   firrtl.subfield %<a_val>[b]       -- descend into inner field
//   dbg.value ...                     -- leaf with meta
//   dbg.struct {"b": ...}             -- inner struct
//   dbg.struct {"a": ...}             -- outer struct
//   dbg.value ... typeName "MyBundle" -- root metadata wrapper
//   dbg.variable "io", ...            -- root variable

// CHECK-LABEL: firrtl.module @NestedBundleSingleLeafTest
firrtl.circuit "NestedBundleSingleLeafTest" {
  firrtl.module @NestedBundleSingleLeafTest() {
    %io = firrtl.wire : !firrtl.bundle<a: bundle<b: uint<8>>>

    // Leaf at depth-2: io.a.b
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.a.b", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<a: bundle<b: uint<8>>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[a]
    // CHECK:      firrtl.subfield %{{.*}}[b]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      %[[S:.+]] = dbg.struct
    // CHECK:      %[[IO:.+]] = dbg.value %[[S]] typeName "MyBundle" : !dbg.struct
    // CHECK:      dbg.variable "io", %[[IO]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<a: bundle<b: uint<8>>>) -> ()
  }
}

// -----

// Nested vector: vector<vector<uint<8>,2>,2> with a circt_debug_subfield leaf
// for path v[0][1]. Exercises two levels of firrtl.subindex in
// DebugAggregateBuilder, producing nested dbg.array ops with the leaf wrapped
// in dbg.value at position [0][1]. Guards subindex-based leaf lookup through
// vector-of-vector.

// CHECK-LABEL: firrtl.module @NestedVecTest
firrtl.circuit "NestedVecTest" {
  firrtl.module @NestedVecTest() {
    %v = firrtl.wire : !firrtl.vector<vector<uint<8>, 2>, 2>

    // Leaf for v[0][1]: parent="v", name="v[0][1]"
    // CHECK-NOT: circt_debug_subfield
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "v[0][1]", typeName: none = "UInt", parent: none = "v">
      %v : (!firrtl.vector<vector<uint<8>, 2>, 2>) -> ()

    // Root circt_debug_var -- must produce nested dbg.array ops:
    //   outer dbg.array contains two inner dbg.array values;
    //   inner dbg.array at index 0 has its element at index 1 wrapped in
    //   a dbg.value carrying the leaf meta.
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %{{.*}}[0]
    // CHECK:      firrtl.subindex %{{.*}}[1]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.array
    // CHECK:      firrtl.subindex %v[1]
    // CHECK:      dbg.array
    // CHECK:      dbg.array
    // CHECK:      dbg.variable "v"
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "Vec">
      %v : (!firrtl.vector<vector<uint<8>, 2>, 2>) -> ()
  }
}

// -----

// circt_debug_var inside a firrtl.layerblock must lower to dbg.variable
// that remains nested inside the same layerblock (walk is recursive).

// CHECK-LABEL: firrtl.module @LayerBlockDebugVar
firrtl.circuit "LayerBlockDebugVar" {
  firrtl.layer @MyLayer bind {}
  firrtl.module @LayerBlockDebugVar(in %in: !firrtl.uint<8>) {
    firrtl.layerblock @MyLayer {
      // CHECK: firrtl.layerblock @MyLayer {
      %w = firrtl.wire : !firrtl.uint<8>
      firrtl.connect %w, %in : !firrtl.uint<8>, !firrtl.uint<8>
      // CHECK: %[[V:.+]] = dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
      // CHECK: dbg.variable "w", %[[V]] : !dbg.value
      firrtl.int.generic "circt_debug_var"
        <name: none = "w", typeName: none = "UInt">
        %w : (!firrtl.uint<8>) -> ()
    }
  }
}

// -----

// Prefix-collision guard: two variables "io" and "iox" exist in the same
// module. A circt_debug_subfield with parent="io" and name="io.x" must only
// match the variable named exactly "io", not "iox". The boundary check in
// processSubfieldIntrinsic uses `starts_with(parent + ".")` and
// `starts_with(parent + "[")` -- bare `starts_with(parent)` would let "io.x"
// match "iox" (since "io.x".starts_with("io") is true for both "io" and "iox"
// after stripping the dot). This test pins that "iox" never gets the leaf.

// CHECK-LABEL: firrtl.module @PrefixCollisionGuard
firrtl.circuit "PrefixCollisionGuard" {
  firrtl.module @PrefixCollisionGuard() {
    %io  = firrtl.wire : !firrtl.bundle<x: uint<8>>
    %iox = firrtl.wire : !firrtl.uint<8>

    // Leaf belongs to "io", NOT to "iox".
    // CHECK-NOT: circt_debug_subfield
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "io.x", typeName: none = "UInt", parent: none = "io">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()

    // "io" variable: must receive the leaf and produce dbg.value.
    // CHECK:      firrtl.subfield %{{.*}}[x]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()

    // "iox" variable: no leaf should be associated; must NOT produce dbg.struct.
    // CHECK:      %[[X:.+]] = dbg.value %iox typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.variable "iox", %[[X]] : !dbg.value
    // CHECK-NOT:  dbg.struct
    firrtl.int.generic "circt_debug_var"
      <name: none = "iox", typeName: none = "UInt">
      %iox : (!firrtl.uint<8>) -> ()
  }
}

// -----

// Mixed-path regression guard (PASSING): verify that `starts_with` in
// processSubfieldIntrinsic (FIRRTLIntrinsics.cpp ~1412) correctly accepts
// both `parent + "["` (vec-rooted leaf) and `parent + "."` (bundle-rooted leaf).
//
// (a) vector<bundle<x: uint<8>>, 2>  with leaf "v[0].x"
//     parent="v", name starts with "v[" -> starts_with(parent+"[") fires. PASSING.
// (b) bundle<v: vector<uint<8>, 1>>  with leaf "b.v[0]"
//     parent="b", name starts with "b." -> starts_with(parent+".") fires. PASSING.
//     (size-1 vector keeps dbg.array operands homogeneous to avoid type mismatch)
//
// If either check regresses, processSubfieldIntrinsic emits a diagnostic and
// firrtl-lower-intrinsics fails, making this FileCheck run fail. Guards:
//   (a) starts_with(parent+"[") still matches vec-rooted paths
//   (b) starts_with(parent+".") still matches bundle-rooted vec paths

// CHECK-LABEL: firrtl.module @MixedPathSubfieldTest
firrtl.circuit "MixedPathSubfieldTest" {
  firrtl.module @MixedPathSubfieldTest() {

    // (a) vector<bundle<x: uint<8>>, 2>, leaf "v[0].x"
    // CHECK-NOT: circt_debug_subfield
    %v = firrtl.wire : !firrtl.vector<bundle<x: uint<8>>, 2>
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "v[0].x", typeName: none = "UInt", parent: none = "v">
      %v : (!firrtl.vector<bundle<x: uint<8>>, 2>) -> ()

    // CHECK:      firrtl.subindex %{{.*}}[0]
    // CHECK:      firrtl.subfield %{{.*}}[x]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.variable "v"
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "VecBundle">
      %v : (!firrtl.vector<bundle<x: uint<8>>, 2>) -> ()

    // (b) bundle<v: vector<uint<8>, 1>>, leaf "b.v[0]"
    // Use size-1 vector so all dbg.array operands share one type (dbg.value),
    // avoiding the mixed-type dbg.array constraint that would fire with size>1
    // when only one element carries a dbg.value leaf.
    %b = firrtl.wire : !firrtl.bundle<v: vector<uint<8>, 1>>
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "b.v[0]", typeName: none = "UInt", parent: none = "b">
      %b : (!firrtl.bundle<v: vector<uint<8>, 1>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[v]
    // CHECK:      firrtl.subindex %{{.*}}[0]
    // CHECK:      dbg.value %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.variable "b"
    firrtl.int.generic "circt_debug_var"
      <name: none = "b", typeName: none = "BundleVec">
      %b : (!firrtl.bundle<v: vector<uint<8>, 1>>) -> ()
  }
}

// -----

// sint<8> operand: the signedness path must lower to dbg.variable just like
// uint<8>. (FIRRTLBaseType::isGround() returns true for sint, so the scalar
// branch in DebugAggregateBuilder fires.)

// CHECK-LABEL: firrtl.module @SIntVar
firrtl.circuit "SIntVar" {
  firrtl.module @SIntVar(in %s: !firrtl.sint<8>) {
    // CHECK: %[[V:.+]] = dbg.value %s typeName "SInt" : !firrtl.sint<8>
    // CHECK: dbg.variable "s", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "s", typeName: none = "SInt">
      %s : (!firrtl.sint<8>) -> ()
  }
}

// -----

// uint<0> (zero-width): still a ground type -> dbg.variable is emitted.

// CHECK-LABEL: firrtl.module @ZeroWidthVar
firrtl.circuit "ZeroWidthVar" {
  firrtl.module @ZeroWidthVar(in %z: !firrtl.uint<0>) {
    // CHECK: %[[V:.+]] = dbg.value %z typeName "UInt" : !firrtl.uint<0>
    // CHECK: dbg.variable "z", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "z", typeName: none = "UInt">
      %z : (!firrtl.uint<0>) -> ()
  }
}

// -----

// 0-operand circt_debug_var resolving to a firrtl.node.

// CHECK-LABEL: firrtl.module @ZeroOperandVarNode
firrtl.circuit "ZeroOperandVarNode" {
  firrtl.module @ZeroOperandVarNode(in %in: !firrtl.uint<8>) {
    // CHECK: %mynode = firrtl.node
    %mynode = firrtl.node %in : !firrtl.uint<8>
    // CHECK: %[[V:.+]] = dbg.value %mynode typeName "UInt" : !firrtl.uint<8>
    // CHECK: dbg.variable "mynode", %[[V]] : !dbg.value
    firrtl.int.generic "circt_debug_var"
      <name: none = "mynode", typeName: none = "UInt"> : () -> ()
  }
}

// -----

// Width-aware enumdef: optional `width` sizes variant IntegerAttrs (here
// uint<2> -> i2). Without `width`, the i64 fallback keeps existing tests.

// CHECK-LABEL: firrtl.module @EnumDefWithWidth
firrtl.circuit "EnumDefWithWidth" {
  firrtl.module @EnumDefWithWidth() {
    // Width-aware parsing still runs (i2 variants), but with no use site the
    // staged data is never materialised into an op, so nothing is emitted.
    // CHECK-NOT: dbg.enum
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "AluOp", fqn: none = "pkg.AluOp$",
       width: i64 = 2,
       variants: none = "[{\"name\":\"ADD\",\"value\":\"0\"},{\"name\":\"SUB\",\"value\":\"1\"},{\"name\":\"AND\",\"value\":\"2\"},{\"name\":\"OR\",\"value\":\"3\"}]">
      : () -> ()
  }
}

// -----

// Pins that width=64 variants with value >= 2^63 (upper unsigned half) parse.

// CHECK-LABEL: firrtl.module @EnumDefWideVariantTest
firrtl.circuit "EnumDefWideVariantTest" {
  firrtl.module @EnumDefWideVariantTest() {
    // Parsing of width=64 variants >= 2^63 still runs; no use site, no op.
    // CHECK-NOT: dbg.enum
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "WideEnum", fqn: none = "pkg.WideEnum$",
       width: i64 = 64,
       variants: none = "[{\"name\":\"ZERO\",\"value\":\"0\"},{\"name\":\"TOP\",\"value\":\"9223372036854775808\"}]">
      : () -> ()
  }
}
