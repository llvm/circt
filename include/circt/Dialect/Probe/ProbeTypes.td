//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBETYPES_TD
#define CIRCT_DIALECT_PROBE_PROBETYPES_TD

include "circt/Dialect/Probe/ProbeDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

class ProbeTypeDef<string name> : TypeDef<ProbeDialect, name> { }

def ProbeElementType : Type<CPred<
    "::circt::hw::isValidProbeElementType($_self)">,
    "type supported for probe observation">;

def RefType : ProbeTypeDef<"Ref"> {
  let mnemonic = "ref";
  let summary = "read-only hardware probe handle";
  let description = [{
    A `probe.ref` type is an SSA handle for read-only observation of a hardware
    value. The handle does not encode or require a particular hierarchical path.
    Its payload type must support probe observation.
  }];

  let parameters = (ins ProbeElementType:$elementType);

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

#endif // CIRCT_DIALECT_PROBE_PROBETYPES_TD
