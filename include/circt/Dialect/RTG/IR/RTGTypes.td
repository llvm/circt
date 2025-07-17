//===- RTGTypes.td - RTG types -----------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the RTG types.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
#define CIRCT_DIALECT_RTG_IR_RTGTYPES_TD

include "circt/Dialect/RTG/IR/RTGDialect.td"
include "mlir/IR/AttrTypeBase.td"

class RTGTypeDef<string name, list<Trait> traits = []>
  : TypeDef<RTGDialect, name, traits>;

//===----------------------------------------------------------------------===//
// Sequence Types
//===----------------------------------------------------------------------===//

def SequenceType : RTGTypeDef<"Sequence"> {
  let summary = "handle to a sequence or sequence family";
  let description = [{
    An SSA value of this type refers to a sequence if the list of element types
    is empty or a sequence family if there are elements left to be substituted.
  }];

  let parameters = (ins OptionalArrayRefParameter<
                          "mlir::Type", "element types">:$elementTypes);

  let mnemonic = "sequence";
  let assemblyFormat = "(`<` $elementTypes^ `>`)?";
}

def RandomizedSequenceType : RTGTypeDef<"RandomizedSequence"> {
  let summary = "handle to a fully randomized sequence";
  let description = [{
    Sequences can contain operations to randomize their content in various ways.
    A sequence of this type is guaranteed to not have any such operations
    anymore (transitively).
  }];

  let mnemonic = "randomized_sequence";
}

def FullySubstitutedSequenceType : DialectType<RTGDialect,
    CPred<"llvm::isa<rtg::SequenceType>($_self) && "
          "llvm::cast<rtg::SequenceType>($_self).getElementTypes().empty()">,
    "fully substituted sequence type", "::circt::rtg::SequenceType">,
  BuildableType<
    "::circt::rtg::SequenceType::get($_builder.getContext(), " #
                                    "llvm::ArrayRef<::mlir::Type>{})">;

//===----------------------------------------------------------------------===//
// Types for common datastructures
//===----------------------------------------------------------------------===//

def SetType : RTGTypeDef<"Set"> {
  let summary = "a set of values";
  let description = [{
    This type represents a standard set datastructure. It does not make any
    assumptions about the underlying implementation. Thus a hash set, tree set,
    etc. can be used in a backend.
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let mnemonic = "set";
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType),
      "return $_get(elementType.getContext(), elementType);">,
  ];
}

def BagType : RTGTypeDef<"Bag"> {
  let summary = "a bag of values";
  let description = [{
    This type represents a standard bag/multiset datastructure. It does not make
    any assumptions about the underlying implementation.
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let mnemonic = "bag";
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType),
      "return $_get(elementType.getContext(), elementType);">,
  ];
}

class SetTypeOf<Type elementType> : ContainerType<
  elementType, SetType.predicate, 
  "llvm::cast<rtg::SetType>($_self).getElementType()", "set">;

class BagTypeOf<Type elementType> : ContainerType<
  elementType, BagType.predicate,
  "llvm::cast<rtg::BagType>($_self).getElementType()", "bag">;

def DictType : RTGTypeDef<"Dict"> {
  let summary = "a dictionary";
  let description = [{
    This type is a dictionary with a static set of entries. This datatype does
    not make any assumptions about how the values are stored (could be a struct,
    a map, etc.). Furthermore, two values of this type should be considered
    equivalent if they have the same set of entry names and types and the values
    match for each entry, independent of the order.
  }];

  let parameters = (ins
    ArrayRefParameter<"::circt::rtg::DictEntry", "dict entries">:$entries);

  let extraClassDeclaration = [{
    /// Checks if the types of the dictionary entries match the ones in the
    /// given type range. The dictionary entries are sorted by ascending names.
    bool entryTypesMatch(mlir::TypeRange types) const;
  }];

  let mnemonic = "dict";

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

def ArrayType : RTGTypeDef<"Array"> {
  let summary = "an array type with dynamic size";
  let description = [{
    Represents an array type with dynamic size. The array contains elements of
    the specified type only.
  }];

  let parameters = (ins "::mlir::Type":$elementType);

  let mnemonic = "array";
  let assemblyFormat = "`<` $elementType `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "mlir::Type":$elementType),
      "return $_get(elementType.getContext(), elementType);">,
  ];
}

def TupleType : RTGTypeDef<"Tuple"> {
  let summary = "a tuple of zero or more fields";
  let description = [{
    This type represents a tuple of zero or more fields. The fields can be
    of any type. The builtin tuple type is not used because it does not allow
    zero fields.
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"::mlir::Type", "tuple field types">:$fieldTypes);

  let mnemonic = "tuple";
  let assemblyFormat = "(`<` $fieldTypes^ `>`)?";
}

//===----------------------------------------------------------------------===//
// Types for ISA targets
//===----------------------------------------------------------------------===//

class RTGISATypeDef<string name, string mnemo, list<Trait> traits = []>
  : RTGTypeDef<name, traits> { let mnemonic = "isa." # mnemo; }


def LabelType : RTGISATypeDef<"Label", "label"> {
  let summary = "a reference to a label";
  let description = [{
    This type represents a label. Payload dialects can add operations to cast
    from this type to an immediate type they can use as an operand to an
    instruction or allow an operand of this type directly.
  }];

  let assemblyFormat = "";
}

def ImmediateType : RTGISATypeDef<"Immediate", "immediate"> {
  let summary = "an ISA immediate";
  let description = [{
    This type represents immediates of arbitrary but fixed bit-width.
    The RTG dialect provides this type to avoid duplication in ISA payload
    dialects.
  }];

  let parameters = (ins "uint32_t":$width);
  let assemblyFormat = "`<` $width `>`";
}

class ImmediateOfWidth<int width> : Type<
  And<[CPred<"llvm::isa<::circt::rtg::ImmediateType>($_self)">,
       CPred<"llvm::cast<::circt::rtg::ImmediateType>($_self).getWidth() == " #
             width>]>,
  "a " # width # "-bit immediate">,
  BuildableType<"::circt::rtg::ImmediateType::get($_builder.getContext(), " #
                width # ")">;

def MemoryType : RTGISATypeDef<"Memory", "memory"> {
  let summary = "handle to a memory";
  let description = [{
    This type is used to represent memory resources that are allocated from
    memory blocks and can be accessed and manipulated by payload dialect
    operations.
  }];

  let parameters = (ins "uint32_t":$addressWidth);
  let assemblyFormat = "`<` $addressWidth `>`";
}

def MemoryBlockType : RTGISATypeDef<"MemoryBlock", "memory_block"> {
  let summary = "handle to a memory block";
  let description = [{
    A memory block is representing a continuous region in a memory map with a
    fixed size and base address. It can refer to actual memory or a memory
    mapped device.

    It is assumed that there is only a single address space.
  }];

  let parameters = (ins "uint32_t":$addressWidth);
  let assemblyFormat = "`<` $addressWidth `>`";
}

def ISAMemory : AnyTypeOf<[MemoryType, ImmediateType, LabelType]>;

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
