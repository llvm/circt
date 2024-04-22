//===- SMTToZ3LLVM.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_SMTTOZ3LLVM_H
#define CIRCT_CONVERSION_SMTTOZ3LLVM_H

#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace circt {

#define GEN_PASS_DECL_LOWERSMTTOZ3LLVM
#include "circt/Conversion/Passes.h.inc"

/// A symbol cache for LLVM globals and functions relevant to SMT lowering
/// patterns.
struct SMTGlobalsHandler {
  /// Creates the LLVM global operations to store the pointers to the solver and
  /// the context and returns a 'SMTGlobalHandler' initialized with those new
  /// globals.
  static SMTGlobalsHandler create(OpBuilder &builder, ModuleOp module);

  /// Initializes the caches and keeps track of the given globals to store the
  /// pointers to the SMT solver and context. It is assumed that the passed
  /// global operations are of the correct (or at least compatible) form. E.g.,
  /// ```
  /// llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  ///   %0 = llvm.mlir.zero : !llvm.ptr
  ///   llvm.return %0 : !llvm.ptr
  /// }
  /// ```
  SMTGlobalsHandler(ModuleOp module, mlir::LLVM::GlobalOp solver,
                    mlir::LLVM::GlobalOp ctx);

  /// Initializes the caches and keeps track of the given globals to store the
  /// pointers to the SMT solver and context. It is assumed that the passed
  /// global operations are of the correct (or at least compatible) form. E.g.,
  /// ```
  /// llvm.mlir.global internal @ctx() {alignment = 8 : i64} : !llvm.ptr {
  ///   %0 = llvm.mlir.zero : !llvm.ptr
  ///   llvm.return %0 : !llvm.ptr
  /// }
  /// ```
  SMTGlobalsHandler(Namespace &&names, mlir::LLVM::GlobalOp solver,
                    mlir::LLVM::GlobalOp ctx);

  /// The global storing the pointer to the SMT solver object currently active.
  const mlir::LLVM::GlobalOp solver;

  /// The global storing the pointer to the SMT context object currently active.
  const mlir::LLVM::GlobalOp ctx;

  Namespace names;
  DenseMap<StringAttr, mlir::LLVM::LLVMFuncOp> funcMap;
  DenseMap<Block *, Value> ctxCache;
  DenseMap<Block *, Value> solverCache;
  DenseMap<StringAttr, mlir::LLVM::GlobalOp> stringCache;
};

/// Populate the given type converter with the SMT to LLVM type conversions.
void populateSMTToZ3LLVMTypeConverter(TypeConverter &converter);

/// Add the SMT to LLVM IR conversion patterns to 'patterns'. A
/// 'SMTGlobalHandler' object has to be passed which acts as a symbol cache for
/// LLVM globals and functions.
void populateSMTToZ3LLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &converter,
    SMTGlobalsHandler &globals, const LowerSMTToZ3LLVMOptions &options);

} // namespace circt

#endif // CIRCT_CONVERSION_SMTTOZ3LLVM_H
