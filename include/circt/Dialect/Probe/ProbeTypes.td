//===- ProbeTypes.td - Probe dialect types ------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBETYPES_TD
#define CIRCT_DIALECT_PROBE_PROBETYPES_TD

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Probe/ProbeDialect.td"
include "circt/Dialect/Seq/SeqTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class ProbeTypeDef<string name> : TypeDef<ProbeDialect, name> { }

def RefType : ProbeTypeDef<"Ref"> {
  let mnemonic = "ref";
  let summary = "read-only hardware probe handle";
  let description = [{
    A `probe.ref` type is an SSA handle for observing a hardware value without
    eagerly resolving the handle to a hierarchical path.  Its payload type is a
    non-inout HW value type.
  }];

  let parameters = (ins "::mlir::Type":$elementType);
  let genVerifyDecl = 1;

  let assemblyFormat = "`<` $elementType `>`";
  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType), [{
      return $_get(elementType.getContext(), elementType);
    }]>
  ];
}

def RefTypeConstraint : DialectType<ProbeDialect,
    CPred<"::llvm::isa<::circt::probe::RefType>($_self)">,
    "probe ref type", "::circt::probe::RefType">;

class SpecificRefType<Type type> : ConfinedType<RefTypeConstraint,
  [SubstLeaves<"$_self",
               "::llvm::cast<::circt::probe::RefType>($_self).getElementType()",
               type.predicate>],
  "ref of " # type.summary, "::circt::probe::RefType"
> {
  Type elementType = type;
}

def ArrayRefType : SpecificRefType<ArrayType>;

def ProbeElementType : Type<CPred<
    "::circt::probe::isProbeElementType($_self)">,
    "probe element type">;

#endif // CIRCT_DIALECT_PROBE_PROBETYPES_TD
