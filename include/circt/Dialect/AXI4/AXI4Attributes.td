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

// An address, printed in hex. Hex is already accepted when parsing.
class AddressParameter<string desc = ""> :
    AttrOrTypeParameter<"uint64_t", desc> {
  let printer = [{
    $_printer << "0x" << ::llvm::utohexstr($_self, /*LowerCase=*/true)
  }];
}

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
    sorted on construction, with the specs of each kind merged into the longest
    of them.
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
      // A `len` is a maximum, so only the longest burst of each kind is kept
      // (and we've already sorted so the longest burst, which is the only one
      // we care about, is at the end)
      ::llvm::SmallVector<::circt::axi4::BurstSpecAttr> merged;
      for (size_t i = 0, e = sorted.size(); i != e; ++i)
        if (i + 1 == e || sorted[i + 1].getKind() != sorted[i].getKind())
          merged.push_back(sorted[i]);
      return $_get($_ctxt, merged);
    }]>
  ];
}

def WindowAttr : AXI4AttrDef<"Window"> {
  let mnemonic = "window";
  let summary = "Describes an access window supported by an endpoint";
  let description = [{
    Consists of a base address, an inclusive last address, and the burst specs
    supported within the window, e.g.
    `#axi4.window<base = 0x4000, last = 0x40ff, burst_specs = <<fixed, len = 4>>>`.
  }];

  let parameters = (ins AddressParameter<>:$base,
                        AddressParameter<>:$last,
                        "::circt::axi4::BurstSetAttr":$burstSpecs);

  let assemblyFormat = [{
    `<` `base` `=` $base `,` `last` `=` $last
        `,` `burst_specs` `=` $burstSpecs `>`
  }];

  let genVerifyDecl = 1;
}

def WindowSetAttr : AXI4AttrDef<"WindowSet"> {
  let mnemonic = "window_set";
  let summary = "Describes a non-empty set of access windows supported by an endpoint";
  let description = [{
    A non-empty set of access windows, e.g.
    `#axi4.window_set<<base = 0x0, last = 0xff, burst_specs = <<fixed, len = 4>>>>`.

    The capabilities at an address are the union of those of every window
    covering it, so windows may overlap. To avoid functionally equivalent types
    being structurally unequal, the windows are normalized on construction into
    the maximal disjoint windows describing the same capabilities, sorted by
    base address. Gaps between windows are preserved.
  }];

  let parameters = (ins
      OptionalArrayRefParameter<"::circt::axi4::WindowAttr">:$windows);

  // Parse empty sets - catch them in the verifier to avoid a very opaque error
  let assemblyFormat = "`<` (`>`) : ($windows^ `>`)?";

  // Force all clients through the normalizing builder below.
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;

  let builders = [
    AttrOrTypeBuilder<(ins
        "::llvm::ArrayRef<::circt::axi4::WindowAttr>":$windows), [{
      return $_get($_ctxt, WindowSetAttr::normalize($_ctxt, windows));
    }]>
  ];

  let extraClassDeclaration = [{
    /// Rewrites a set of possibly overlapping windows into an equivalent
    /// ordered and uniqued set of disjoint windows
    static ::llvm::SmallVector<WindowAttr>
    normalize(::mlir::MLIRContext *ctx, ::llvm::ArrayRef<WindowAttr> windows);
  }];
}

#endif // CIRCT_DIALECT_AXI4_AXI4ATTRIBUTES_TD
