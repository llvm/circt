//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the attributes for the AXI4 dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD
#define CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD

include "circt/Dialect/AXI4/AXI4Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Enums
//===----------------------------------------------------------------------===//

def BurstKindFixed : I32EnumAttrCase<"Fixed", 0, "fixed">;
def BurstKindIncr  : I32EnumAttrCase<"Incr", 1, "incr">;
def BurstKindWrap  : I32EnumAttrCase<"Wrap", 2, "wrap">;

def BurstKind : I32EnumAttr<"BurstKind", "AXI4 burst kind",
    [BurstKindFixed, BurstKindIncr, BurstKindWrap]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::axi4";
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

class AXI4AttrDef<string name, list<Trait> traits = []> :
    AttrDef<AXI4Dialect, name, traits>;

def BurstSpecAttr : AXI4AttrDef<"BurstSpec"> {
  let mnemonic = "burst_spec";
  let summary = "Describes the burst capabilities of an endpoint";
  let description = [{
    Consists of a burst kind and a maximum burst length in beats, e.g.
    `#axi4.burst_spec<fixed, len = 4>` or `#axi4.burst_spec<incr, len = 256>`.

    AXI4 permits 1-16 beats for `fixed`, 1-256 for `incr`, and only 2, 4, 8, or
    16 for `wrap`.
  }];

  let parameters = (ins EnumParameter<BurstKind>:$kind,
                        "uint32_t":$len);

  let assemblyFormat = "`<` $kind `,` `len` `=` $len `>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Ordering over burst_specs for canonical typing
    static bool compareCanonical(BurstSpecAttr lhs, BurstSpecAttr rhs);
  }];
}

def BurstSetAttr : AXI4AttrDef<"BurstSet"> {
  let mnemonic = "burst_set";
  let summary = "Describes a non-empty set of burst specs supported by an endpoint";
  let description = [{
    A non-empty set of burst specs, e.g. `#axi4.burst_set<<fixed, len = 4>>` or
    `#axi4.burst_set<<fixed, len = 4>, <incr, len = 8>>`.

    To avoid functionally equivalent types (e.g. types containing burst_sets with
    the same specs in different orders) being structurally inequivalent, this is
    sorted and de-duplicated on construction.
  }];

  let parameters = (ins
      OptionalArrayRefParameter<"::circt::axi4::BurstSpecAttr">:$burstSpecs);

  // Parse empty sets - catch them in the verifier to avoid a very opaque error
  let assemblyFormat = "`<` (`>`) : ($burstSpecs^ `>`)?";

  // Force all clients through the canonicalizing builder below.
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    AttrOrTypeBuilder<(ins
        "::llvm::ArrayRef<::circt::axi4::BurstSpecAttr>":$burstSpecs), [{
      ::llvm::SmallVector<::circt::axi4::BurstSpecAttr> sorted(
          burstSpecs.begin(), burstSpecs.end());
      ::llvm::sort(sorted, BurstSpecAttr::compareCanonical);
      sorted.erase(::llvm::unique(sorted), sorted.end());
      return $_get($_ctxt, sorted);
    }]>
  ];
}

#endif // CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD
