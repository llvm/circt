//===- StdWrapper.h - Standard wrapper ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the definition of the StdWrapper class, an HLT wrapper for
// wrapping builtin.func based kernels.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_HLT_WRAPGEN_STD_STDWRAPPER_H
#define CIRCT_TOOLS_HLT_WRAPGEN_STD_STDWRAPPER_H

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "circt/Tools/hlt/WrapGen/BaseWrapper.h"
#include "circt/Tools/hlt/WrapGen/CEmitterUtils.h"

using namespace llvm;
using namespace mlir;

namespace circt {
namespace hlt {

class StdWrapper : public BaseWrapper {
public:
  using BaseWrapper::BaseWrapper;
  LogicalResult init(Operation *refOp, Operation *kernelOp);
  LogicalResult emitPreamble(Operation *kernelOp) override;

protected:
  SmallVector<std::string> getIncludes() override;
  SmallVector<std::string> getNamespaces() override { return {"circt", "hlt"}; }
};

} // namespace hlt
} // namespace circt

#endif // CIRCT_TOOLS_HLT_WRAPGEN_STD_STDWRAPPER_H
