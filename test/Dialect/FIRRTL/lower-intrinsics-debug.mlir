// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-lower-intrinsics)))' %s --split-input-file | FileCheck %s

// Scalar, bundle, and vec cases for the three Chisel debug intrinsics
// (circt_debug_moduleinfo, circt_debug_var, circt_debug_subfield) and
// circt_debug_enumdef.

// CHECK-LABEL: firrtl.module @DebugTest
firrtl.circuit "DebugTest" {
  firrtl.module @DebugTest(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {

    // 1. moduleinfo (empty params array is forwarded as [])
    // CHECK-DAG: dbg.moduleinfo typeName "MyModule" params []
    firrtl.int.generic "circt_debug_moduleinfo"
      <typeName: none = "MyModule", params: none = "[]"> : () -> ()

    // 2. enumdef with JSON array (value serialized as string by frontend)
    // CHECK-DAG: dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
    firrtl.int.generic "circt_debug_enumdef"
      <typeName: none = "MyState", fqn: none = "pkg.MyState$",
       variants: none = "[{\"name\":\"Idle\",\"value\":\"0\"},{\"name\":\"Run\",\"value\":\"1\"}]">
      : () -> ()

    // 3. circt_debug_var on scalar wire -> dbg.variable with the wire value
    // CHECK: dbg.variable "w", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    %w = firrtl.wire : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "w", typeName: none = "UInt">
      %w : (!firrtl.uint<8>) -> ()

    // 4. circt_debug_var on a passive bundle -> dbg.struct + dbg.variable
    //    The converter builds:
    //      %a = firrtl.subfield %io[in]
    //      %b = firrtl.subfield %io[out]
    //      %s = dbg.struct {"in": %a, "out": %b}
    //      dbg.variable "io", %s
    // CHECK:      firrtl.subfield %io[in]
    // CHECK:      firrtl.subfield %io[out]
    // CHECK:      dbg.struct
    // CHECK: dbg.variable "io", %{{.*}} typeName "MyBundle"
    %io = firrtl.wire : !firrtl.bundle<in: uint<8>, out: uint<8>>
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<in: uint<8>, out: uint<8>>) -> ()

    // 5. circt_debug_var on a Vec -> dbg.array + dbg.variable
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %v[1]
    // CHECK:      dbg.array
    // CHECK-NEXT: dbg.variable "v", %{{.*}} typeName "Vec"
    %v = firrtl.wire : !firrtl.vector<uint<4>, 2>
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "Vec">
      %v : (!firrtl.vector<uint<4>, 2>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with an enum-typed field -- the field must get a dbg.subfield
// wrapping it with an enumDef reference.
//
// Chisel emits:
//   circt_debug_enumdef for MyState
//   circt_debug_subfield(parent="io", name="state") for io.state with enumFqn
//   circt_debug_subfield(parent="io", name="data")  for io.data
//   circt_debug_var(name="io")                      for io

// CHECK-LABEL: firrtl.module @BundleEnumFieldTest
firrtl.circuit "BundleEnumFieldTest" {
  firrtl.module @BundleEnumFieldTest(
      in %in: !firrtl.uint<8>,
      out %out: !firrtl.uint<8>) {

    // enumdef must appear in the output
    // CHECK: %[[E:.+]] = dbg.enumdef "MyState", fqn "pkg.MyState$", {Idle = 0 : i64, Run = 1 : i64}
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

    // Root circt_debug_var -- produces dbg.struct + dbg.variable
    // CHECK:      firrtl.subfield %{{.*}}[state]
    // CHECK:      dbg.subfield "io.state", %{{.*}} typeName "UInt" enumDef %[[E]] : !firrtl.uint<2>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.subfield "io.data", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io", %{{.*}} typeName "MyBundle"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<state: uint<2>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// Bundle with a FixedPoint field carrying type parameters.
// Verifies that the "params" JSON string is parsed and forwarded to
// dbg.subfield as an ArrayAttr of DictionaryAttrs.

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
    // CHECK:      dbg.subfield "io.fp", %{{.*}} typeName "FixedPoint" params [{name = "width", value = "8"}, {name = "binaryPoint", value = "4"}] : !firrtl.uint<8>
    // CHECK:      firrtl.subfield %{{.*}}[data]
    // CHECK:      dbg.subfield "io.data", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io", %{{.*}} typeName "MyBundle"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<fp: uint<8>, data: uint<8>>) -> ()

    firrtl.connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

// -----

// moduleinfo with constructor params

// CHECK-LABEL: firrtl.module @ModuleInfoWithParams
firrtl.circuit "ModuleInfoWithParams" {
  firrtl.module @ModuleInfoWithParams() {
    // CHECK: dbg.moduleinfo typeName "Counter" params [{name = "width", value = "8"}]
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
    // CHECK: dbg.variable "a", %a typeName "UInt" : !firrtl.uint<8>
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
    // CHECK: dbg.variable "mywire", %mywire typeName "UInt" : !firrtl.uint<8>
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
    // CHECK: dbg.variable "myreg", %myreg typeName "UInt" : !firrtl.uint<8>
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
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "r", %{{.*}} typeName "MyBundle" : !dbg.struct
    firrtl.int.generic "circt_debug_var"
      <name: none = "r", typeName: none = "MyBundle"> : () -> ()
  }
}

// -----

// Nested aggregate: bundle whose field is itself a bundle. Exercises
// recursion in buildDebugAggregateWithMeta -- the outer struct contains
// an inner struct, with a leaf-meta entry on the deepest field.

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
    // CHECK:      dbg.subfield "io.inner.a", %{{.*}} typeName "UInt" : !firrtl.uint<4>
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
//   inner struct {"b": dbg.subfield "io.a.b"}
//
// Expected lowering:
//   firrtl.subfield %io[a]            -- descend into outer field
//   firrtl.subfield %<a_val>[b]       -- descend into inner field
//   dbg.subfield "io.a.b", ...        -- leaf with meta
//   dbg.struct {"b": ...}             -- inner struct
//   dbg.struct {"a": ...}             -- outer struct
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
    // CHECK:      dbg.subfield "io.a.b", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io", %{{.*}} typeName "MyBundle"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<a: bundle<b: uint<8>>>) -> ()
  }
}

// -----

// Nested vector: vector<vector<uint<8>,2>,2> with a circt_debug_subfield leaf
// for path v[0][1]. Exercises two levels of firrtl.subindex in
// buildDebugAggregateWithMeta, producing nested dbg.array ops with the leaf
// wrapped in dbg.subfield at position [0][1].
//
// FIXME: if this test fails, the converter does not yet recurse into
// vector-of-vector with subindex-based leaf lookup (nested dbg.array path
// resolution unimplemented).

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
    //   dbg.subfield "v[0][1]".
    // CHECK:      firrtl.subindex %v[0]
    // CHECK:      firrtl.subindex %{{.*}}[0]
    // CHECK:      firrtl.subindex %{{.*}}[1]
    // CHECK:      dbg.subfield "v[0][1]", %{{.*}} typeName "UInt" : !firrtl.uint<8>
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
      // CHECK: dbg.variable "w", %{{.*}} typeName "UInt" : !firrtl.uint<8>
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

    // "io" variable: must receive the leaf and produce dbg.subfield.
    // CHECK:      firrtl.subfield %{{.*}}[x]
    // CHECK:      dbg.subfield "io.x", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.struct
    // CHECK:      dbg.variable "io"
    firrtl.int.generic "circt_debug_var"
      <name: none = "io", typeName: none = "MyBundle">
      %io : (!firrtl.bundle<x: uint<8>>) -> ()

    // "iox" variable: no leaf should be associated; must NOT produce dbg.struct.
    // CHECK:      dbg.variable "iox", %iox
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
// firrtl-lower-intrinsics fails, making this FileCheck run fail.
// FIXME(a): regression if starts_with(parent+"[") stops matching vec-rooted paths
// FIXME(b): regression if starts_with(parent+".") stops matching bundle-rooted vec paths

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
    // CHECK:      dbg.subfield "v[0].x", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.variable "v"
    firrtl.int.generic "circt_debug_var"
      <name: none = "v", typeName: none = "VecBundle">
      %v : (!firrtl.vector<bundle<x: uint<8>>, 2>) -> ()

    // (b) bundle<v: vector<uint<8>, 1>>, leaf "b.v[0]"
    // Use size-1 vector so all dbg.array operands share one type (dbg.subfield),
    // avoiding the mixed-type dbg.array constraint that would fire with size>1
    // when only one element carries a dbg.subfield leaf.
    %b = firrtl.wire : !firrtl.bundle<v: vector<uint<8>, 1>>
    firrtl.int.generic "circt_debug_subfield"
      <name: none = "b.v[0]", typeName: none = "UInt", parent: none = "b">
      %b : (!firrtl.bundle<v: vector<uint<8>, 1>>) -> ()

    // CHECK:      firrtl.subfield %{{.*}}[v]
    // CHECK:      firrtl.subindex %{{.*}}[0]
    // CHECK:      dbg.subfield "b.v[0]", %{{.*}} typeName "UInt" : !firrtl.uint<8>
    // CHECK:      dbg.variable "b"
    firrtl.int.generic "circt_debug_var"
      <name: none = "b", typeName: none = "BundleVec">
      %b : (!firrtl.bundle<v: vector<uint<8>, 1>>) -> ()
  }
}

// -----

// sint<8> operand: the signedness path must lower to dbg.variable just like
// uint<8>. (FIRRTLBaseType::isGround() returns true for sint, so the scalar
// branch in buildDebugAggregateWithMeta fires.)

// CHECK-LABEL: firrtl.module @SIntVar
firrtl.circuit "SIntVar" {
  firrtl.module @SIntVar(in %s: !firrtl.sint<8>) {
    // CHECK: dbg.variable "s", %s typeName "SInt" : !firrtl.sint<8>
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
    // CHECK: dbg.variable "z", %z typeName "UInt" : !firrtl.uint<0>
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
    // CHECK: dbg.variable "mynode", %mynode typeName "UInt" : !firrtl.uint<8>
    firrtl.int.generic "circt_debug_var"
      <name: none = "mynode", typeName: none = "UInt"> : () -> ()
  }
}
