//===- HandshakeToDC.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Handshake dialect to
// CIRCT RTL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HANDSHAKETODC_H
#define CIRCT_CONVERSION_HANDSHAKETODC_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

#define GEN_PASS_DECL_HANDSHAKETODC
#include "circt/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createHandshakeToDCPass();

namespace handshaketodc {
using ConvertedOps = DenseSet<Operation *>;

// Runs Handshake to DC conversion on the provided op. `patternBuilder` can be
// used to describe additional patterns to run - typically this will be a
// pattern that converts the container operation (e.g. `op`).
// `configureTarget` can be provided to specialize legalization.
LogicalResult runHandshakeToDC(
    mlir::Operation *op,
    llvm::function_ref<void(TypeConverter &typeConverter,
                            ConvertedOps &convertedOps,
                            RewritePatternSet &patterns)>
        patternBuilder,
    llvm::function_ref<void(mlir::ConversionTarget &)> configureTarget = {});
} // namespace handshaketodc

namespace handshake {

// Converts 't' into a valid HW type. This is strictly used for converting
// 'index' types into a fixed-width type.
Type toValidType(Type t);

} // namespace handshake
} // namespace circt

#endif // CIRCT_CONVERSION_HANDSHAKETODC_H
