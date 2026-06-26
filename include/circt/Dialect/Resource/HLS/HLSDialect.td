//===- HLSDialect.td - HLS dialect definition --------------*- tablegen -*-===//
//
// The `hls` dialect: makes Vitis-HLS implementation decisions explicit in the
// IR. First op is a write-enable (predicated) store; the dialect is structured
// to grow (resource bindings, banking structure, etc.) as the need arises.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLSDIALECT_TD
#define CIRCT_DIALECT_HLS_HLSDIALECT_TD

include "mlir/IR/OpBase.td"

def HLSDialect : Dialect {
  let name = "hls";
  let summary = "HLS resource and scheduling decisions made explicit.";
  let description = [{
    Ops that carry Vitis-HLS implementation decisions the rest of the pipeline
    would otherwise have to reconstruct: starting with a predicated
    (write-enable) memory write that models a BRAM port whose write-enable is
    driven by a runtime guard, instead of the read-modify-write `arith.select`
    demux that fabricates a phantom read port.
  }];

  let cppNamespace = "circt::hls_analysis";

  // No custom types/attrs yet; flip these on when you add a bind_storage attr.
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypePrinterParser = 0;
}

// Base class for every op in this dialect.
class HLSOp<string mnemonic, list<Trait> traits = []>
    : Op<HLSDialect, mnemonic, traits>;

#endif // CIRCT_DIALECT_HLS_HLSDIALECT_TD
