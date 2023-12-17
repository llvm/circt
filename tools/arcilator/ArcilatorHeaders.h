//===- ArcilatorHeaders.h - Arcilator headers generator -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generates header files to link to the output of the `arcilator` compiler.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_ARCILATOR_ARCILATORHEADERS_H
#define CIRCT_TOOLS_ARCILATOR_ARCILATORHEADERS_H

#include "circt/Dialect/Arc/Transforms/PrintStateInfo.h"
#include "mlir/Support/LogicalResult.h"

namespace circt {
namespace arcilator {

/// Generates in `output` the C++ header file to link to the provided `models`.
mlir::LogicalResult generateHeaders(llvm::raw_ostream &output,
                                    const std::vector<arc::ModelInfo> &models);

} // namespace arcilator
} // namespace circt

#endif // CIRCT_TOOLS_ARCILATOR_ARCILATORHEADERS_H
