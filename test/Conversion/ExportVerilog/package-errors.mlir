// RUN: circt-opt %s -export-verilog -verify-diagnostics -split-input-file -o /dev/null

// A package cannot be emitted into more than one file.
// expected-error @+1 {{packages can only be emitted to a single file}}
sv.package @duplicated {}
emit.file "first.sv" { emit.ref @duplicated }
emit.file "second.sv" { emit.ref @duplicated }

// -----

// External package names cannot be legalized by renaming them.
// expected-error @+1 {{name "module" is not allowed in Verilog output}}
sv.package.extern @module {}

// -----

sv.package.extern @InvalidType {
  // expected-error @+1 {{external package member name "word-type" is not allowed in Verilog output}}
  hw.typedecl @word, "word-type" : i8
}

// -----

sv.package.extern @KeywordType {
  // expected-error @+1 {{external package member name "logic" is not allowed in Verilog output}}
  hw.typedecl @word, "logic" : i8
}

// -----

module attributes {circt.loweringOptions = "caseInsensitiveKeywords"} {
  sv.package.extern @MixedCaseKeyword {
    // expected-error @+1 {{external package member name "Module" is not allowed in Verilog output}}
    hw.typedecl @word, "Module" : i8
  }
}

// -----

sv.package.extern @DuplicateTypes {
  hw.typedecl @first, "Word" : i8
  // expected-error @+1 {{external package member name "Word" is not unique}}
  hw.typedecl @second, "Word" : i8
}

// -----

sv.package.extern @EscapedDuplicate {
  hw.typedecl @word : i8
  // expected-error @+1 {{external package member name "\word" is not unique}}
  hw.typedecl @escaped, "\\word" : i8
}

// -----

sv.package.extern @EnumTypeCollision {
  // expected-error @+1 {{external package member name "State_Idle" is not unique}}
  hw.typedecl @State : !hw.enum<Idle, Busy>
  hw.typedecl @State_Idle : i8
}

// -----

sv.package.extern @EnumMemberCollision {
  hw.typedecl @A_B : !hw.enum<C>
  // expected-error @+1 {{external package member name "A_B_C" is not unique}}
  hw.typedecl @A : !hw.enum<B_C>
}

// -----

sv.package.extern @InvalidEnumMember {
  // expected-error @+1 {{external package member name "State_bad.field" is not allowed in Verilog output}}
  hw.typedecl @State : !hw.enum<bad.field>
}
