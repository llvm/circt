//===- ESIOps.td - ESI dialect operation definitions --------- tablegen -*-===//
//
// This is where any operations for the ESI dialect will go
//
//===----------------------------------------------------------------------===//


class ESI_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ESI_Dialect, mnemonic, traits>;
