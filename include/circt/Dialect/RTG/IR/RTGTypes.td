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

class RTGTypeDef<string name> : TypeDef<RTGDialect, name>;

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

def LabelType : RTGTypeDef<"Label"> {
  let summary = "a reference to a label";
  let description = [{
    This type represents a label. Payload dialects can add operations to cast
    from this type to an immediate type they can use as an operand to an
    instruction.
  }];

  let mnemonic = "label";
  let assemblyFormat = "";
}

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

class ImmTypeBase<int width> : TypeDef<RTGDialect, "Imm" # width, []> {
  let summary = "represents a " # width # "-bit immediate";
  let mnemonic = "imm" # width;
}

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
