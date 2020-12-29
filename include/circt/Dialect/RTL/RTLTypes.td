//===- RTLTypes.td - RTL data type definitions -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data types for the RTL dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class RTLType<string name> : TypeDef<RTLDialect, name> { }

// A simple fixed size array.
def ArrayType : RTLType<"Array"> {
  let summary = "fixed-sized array";
  let description = [{
    Fixed sized RTL arrays are roughly similar to C arrays. On the wire (vs.
    in a memory), arrays are always packed. Memory layout is not defined as
    it does not need to be since in silicon there is not implicit memory
    sharing.
  }];

  let mnemonic = "array";
  let parameters = (ins "::mlir::Type":$elementType, "size_t":$size);
  let genVerifyInvariantsDecl = 1;

  let extraClassDeclaration = [{
    static ArrayType get(Type elementType, size_t size) {
      return get(elementType.getContext(), elementType, size);
    }
  }];
}

def InOutType : RTLType<"InOut"> {
  let summary = "inout type";
  let description = [{
    InOut type is used for model operations and values that have "connection"
    semantics, instead of typical dataflow behavior.  This is used for wires
    and inout ports in Verilog.
  }];

  let mnemonic = "inout";
  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyInvariantsDecl = 1;

  let extraClassDeclaration = [{
    static InOutType get(Type elementType) {
      return get(elementType.getContext(), elementType);
    }
  }];
}

// Type constraint that indicates that an operand/result may only be a valid
// inout element type.
def InOutElementType : Type<CPred<"isRTLValueType($_self)">,
                            "a valid inout element">;

// This type constraint is used by operations whose asm strings have a type "T"
// in the asmstring, which constraints an input or output to type `T`, and then
// constraints another to `!rtl.inout<T>`.  The "value" and "inoutValue" are the
// name of the operand or result for the type sides of this.
//
// Note that the inoutValue must be constrained to being an InOutElementType to
// ensure that functions and other things aren't passed into this.
class InOutTypeConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   value, inoutValue,
                   "InOutType::get($_self.getContext(), $_self)">;

// This type constraint indicates that the specified value matches the element
// type of the specified inoutValue.  It is the dual to InOutTypeConstraint used
// when inference needs to "go the other way".
//
// Note that the inoutValue must be constrained to being an InOutElementType to
// ensure that functions and other things aren't passed into this.
class InOutElementConstraint<string value, string inoutValue>
  : TypesMatchWith<"type should be element of inout type",
                   inoutValue, value,
                   "$_self.cast<InOutType>().getElementType()">;

// Adapted from TestTypeDefs.td
// A parent type for any type which is just a list of fields (e.g. structs,
// unions).
class FieldInfo_Type<string name> : RTLType<name> {
  let parameters = (
    ins
    // An ArrayRef of something which requires allocation in the storage
    // constructor.
    ArrayRefOfSelfAllocationParameter<
      "::circt::rtl::FieldInfo",
      "Models struct fields">: $fields
  );

  // Prints the type in this format:
  //   struct<[{field1Name, field1Type}, {field2Name, field2Type}]>
  let printer = [{
    $_printer << "struct" << "<";
    for (size_t i=0, e = getImpl()->fields.size(); i < e; i++) {
      const auto& field = getImpl()->fields[i];
      $_printer << "{" << field.name << "," << field.type << "}";
      if (i < getImpl()->fields.size() - 1)
          $_printer << ",";
    }
    $_printer << ">";
  }];

  // Parses the above format
  let parser = [{
    llvm::SmallVector<FieldInfo, 4> parameters;
    if ($_parser.parseLess()) return Type();
    while (mlir::succeeded($_parser.parseOptionalLBrace())) {
      StringRef name;
      if ($_parser.parseKeyword(&name)) return Type();
      if ($_parser.parseComma()) return Type();
      Type type;
      if ($_parser.parseType(type)) return Type();
      if ($_parser.parseRBrace()) return Type();
      parameters.push_back(FieldInfo {name, type});
      if ($_parser.parseOptionalComma()) break;
    }
    if ($_parser.parseGreater()) return Type();
    return get($_ctxt, parameters);
  }];
}

def StructType : FieldInfo_Type<"Struct"> {
  let summary = "struct type for ports and wires";
  let description = [{
    Struct type is used...
  }];
    let mnemonic = "struct";
}

// This type constraint is used to constraing the value type to the type of the 
// named filed of the structValue
class StructFieldTypeConstraint<string structValue, string value, string fieldName>
  : TypesMatchWith<"type should be element of struct type",
                   structValue, value,
      "getTypeByField($_self, $fieldName)">;
