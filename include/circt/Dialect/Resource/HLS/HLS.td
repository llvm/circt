//===- HLS.td - HLS dialect umbrella -----------------------*- tablegen -*-===//
//
// This is the file `add_circt_dialect(HLS hls)` points LLVM_TARGET_DEFINITIONS
// at. It pulls in the dialect definition and the op set so the generated
// HLS{,Dialect}.{h,cpp}.inc cover everything.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HLS_HLS_TD
#define CIRCT_DIALECT_HLS_HLS_TD

include "circt/Dialect/Resource/HLS/HLSDialect.td"
include "circt/Dialect/Resource/HLS/HLSOps.td"

#endif // CIRCT_DIALECT_HLS_HLS_TD
