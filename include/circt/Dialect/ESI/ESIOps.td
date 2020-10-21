//===- ESIOps.td - ESI dialect operation definitions --------- tablegen -*-===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//


class ESI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;
