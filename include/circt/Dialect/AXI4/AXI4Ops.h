//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_AXI4_AXI4OPS_H
#define CIRCT_DIALECT_AXI4_AXI4OPS_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "circt/Dialect/AXI4/AXI4Types.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqTypes.h"

namespace circt {
namespace axi4 {
namespace OpTrait {
/// Constrains an op's `!axi4.port` results to at most one use each
template <typename ConcreteType>
class PortResultsAtMostOneUse
    : public mlir::OpTrait::TraitBase<ConcreteType, PortResultsAtMostOneUse> {
public:
  static llvm::LogicalResult verifyTrait(mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      if (mlir::isa<PortType>(result.getType()) && result.hasNUsesOrMore(2))
        return op->emitOpError(
            "port result must have at most one use; route through an "
            "'axi4.xbar' to fan out to multiple endpoints");
    return mlir::success();
  }
};
} // namespace OpTrait
} // namespace axi4
} // namespace circt

#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.h.inc"

#endif // CIRCT_DIALECT_AXI4_AXI4OPS_H
