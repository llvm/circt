//===-- circt-c/Conversion/SeqToSV.h - C API for seq to sv conversion -----===//
//
// This header declares the C interface for registering conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_CONVERSION_SEQTOSV_H
#define CIRCT_C_CONVERSION_SEQTOSV_H

#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void registerConvertSeqToSVPass();

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_CONVERSION_SEQTOSV_H
