//===- TargetOptions.h - CIRCT Lowering Options ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Options for controlling the lowering process and verilog exporting based
// on a target-triple-like format.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_TARGETOPTIONS_H
#define CIRCT_SUPPORT_TARGETOPTIONS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include <variant>

namespace mlir {
class ModuleOp;
}

namespace circt {

struct TargetOptions {
  /// Error callback type used to indicate errors parsing the options string.
  using ErrorHandlerT = llvm::function_ref<void(llvm::Twine)>;

  TargetOptions(mlir::MLIRContext *ctx) : ctx(ctx) {}

  /// Read in options from a string, overriding only the set options in the
  /// string.
  void parse(llvm::StringRef options, ErrorHandlerT errorHandler);

  struct Option {
    using Options = llvm::DenseMap<mlir::StringAttr, Option>;
    mlir::Attribute value;
    std::optional<Options> options;
  };
  using Options = Option::Options;

  /// Looks up the target option at the given path.
  mlir::FailureOr<Option>
  getOption(llvm::SmallVector<llvm::StringRef> path) const;

  /// Like getOption, but dereferences the option under an assumption that it
  /// is an attribute.
  mlir::FailureOr<mlir::Attribute>
  getValue(llvm::SmallVector<llvm::StringRef> path) const;

  Options options;

  mlir::MLIRContext *ctx;
};

} // namespace circt

#endif // CIRCT_SUPPORT_TARGETOPTIONS_H