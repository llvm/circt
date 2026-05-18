// RUN: circt-opt %s --verify-diagnostics --split-input-file

// -----
// dbg.enumdef with a non-IntegerAttr variant value.

func.func @EnumDefNonIntegerAttr() {
  // expected-error @+1 {{variantsMap entry 'A' must be an IntegerAttr}}
  %e = dbg.enumdef "MyState", fqn "pkg.MyState$", {A = "not-an-int"}
  return
}

// -----
// dbg.enumdef with duplicate variant values.

func.func @EnumDefDuplicateValues() {
  // expected-error @+1 {{duplicate enum value 0}}
  %e = dbg.enumdef "MyState", fqn "pkg.MyState$", {A = 0 : i64, B = 0 : i64}
  return
}

// -----
// dbg.subfield used by a non-struct/array op triggers verifier error.

func.func @SubFieldBadUser() {
  %c = arith.constant 0 : i32
  // expected-error @+1 {{must only be used as an operand of dbg.struct or dbg.array}}
  %sf = dbg.subfield "x", %c : i32
  dbg.variable "y", %sf : !dbg.subfield
  return
}

// -----
// dbg.moduleinfo uniqueness: two in the same region must be rejected.

func.func @ModuleInfoDuplicate() {
  dbg.moduleinfo typeName "MyMod"
  // expected-error @+1 {{only one dbg.moduleinfo may appear in a region}}
  dbg.moduleinfo typeName "MyMod"
  return
}

// -----
// dbg.moduleinfo uniqueness must hold across blocks of the same region, not
// just within a single block.

func.func @ModuleInfoDuplicateAcrossBlocks() {
  dbg.moduleinfo typeName "MyMod"
  cf.br ^bb1
^bb1:
  // expected-error @+1 {{only one dbg.moduleinfo may appear in a region}}
  dbg.moduleinfo typeName "MyMod"
  return
}

// -----
// dbg.enumdef with an empty variantsMap is semantically useless and rejected.

func.func @EnumDefEmptyVariants() {
  // expected-error @+1 {{variantsMap must not be empty}}
  %e = dbg.enumdef "Empty", fqn "pkg.Empty", {}
  return
}
