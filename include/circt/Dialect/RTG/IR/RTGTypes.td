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
  let summary = "handle to a sequence closure";
  let description = [{
    An SSA value of this type refers to an `rtg.sequence` operation and the
    argument values it should be invoked with (if it has any).
  }];

  let mnemonic = "sequence";
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

class SetTypeOf<Type elementType> : ContainerType<
  elementType, SetType.predicate, 
  "llvm::cast<rtg::SetType>($_self).getElementType()", "set">;

#endif // CIRCT_DIALECT_RTG_IR_RTGTYPES_TD
