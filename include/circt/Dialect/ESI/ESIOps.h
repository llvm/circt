//===- ESIOps.h - ESI operations --------------------------------*- C++ -*-===//
//
// ESI Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIOPS_H
#define CIRCT_DIALECT_ESI_ESIOPS_H

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

// namespace circt {
// namespace esi {

// /// Control options for ChannelBufferOps.
// struct ChannelBufferOptions {
//   llvm::Optional<size_t> stages; // Pipeline stages.
// };
// } // namespace esi
// } // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.h.inc"

#endif
