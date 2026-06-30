// RUN: circt-opt %s --verify-diagnostics --split-input-file

// -----
// dbg.enum with a non-IntegerAttr variant value.

func.func @EnumNonIntegerAttr() {
  %c = arith.constant 0 : i2
  // expected-error @+1 {{variantsMap entry 'A' must be an IntegerAttr}}
  %e = dbg.enum %c, "MyState", {A = "not-an-int"} : i2
  return
}

// -----
// dbg.enum with duplicate variant values.

func.func @EnumDuplicateValues() {
  %c = arith.constant 0 : i2
  // expected-error @+1 {{duplicate enum value 0}}
  %e = dbg.enum %c, "MyState", {A = 0 : i64, B = 0 : i64} : i2
  return
}

// -----
// dbg.value used by a non-variable/struct/array op triggers verifier error.

func.func @ValueBadUser() {
  %c = arith.constant 0 : i32
  // expected-error @+1 {{must only be used as an operand of dbg.variable, dbg.struct, or dbg.array}}
  %v = dbg.value %c : i32
  %v2 = dbg.value %v : !dbg.value
  return
}

// -----
// dbg.enum with an empty variantsMap is semantically useless and rejected.

func.func @EnumEmptyVariants() {
  %c = arith.constant 0 : i2
  // expected-error @+1 {{variantsMap must not be empty}}
  %e = dbg.enum %c, "Empty", {} : i2
  return
}

// -----
// dbg.enum variant value with signed integer type (si64) must be rejected.

func.func @EnumSignedVariant() {
  %c = arith.constant 0 : i2
  // expected-error @+1 {{variant 'A' must have a signless integer value}}
  %e = dbg.enum %c, "MyState", {A = 0 : si64} : i2
  return
}

// -----
// dbg.enum variant value with unsigned integer type (ui64) must be rejected.

func.func @EnumUnsignedVariant() {
  %c = arith.constant 0 : i2
  // expected-error @+1 {{variant 'A' must have a signless integer value}}
  %e = dbg.enum %c, "MyState", {A = 0 : ui64} : i2
  return
}
