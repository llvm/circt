//===- ESIOps.h - ESI operations --------------------------------*- C++ -*-===//
//
// ESI Ops are defined in tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_ESIOPS_H
#define CIRCT_DIALECT_ESI_ESIOPS_H

#include "circt/Dialect/ESI/ESIDialect.h"
#include "circt/Dialect/ESI/ESITypes.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "circt/Dialect/ESI/ESI.h.inc"

#endif
