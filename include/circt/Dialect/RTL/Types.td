//===- Types.td - RTL data type definitions ----------------*- tablegen -*-===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class RTLType<string name> : TypeDef<RTLDialect, name> { }

// A simple fixed size array.
def ArrayType : RTLType<"Array"> {
  let summary = "A fixed-sized array";
  let description = [{
    Fixed sized RTL arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "array";
  let parameters = (ins "::mlir::Type":$innerType, "size_t":$size);
}
