//===- ExportLLVM.cpp - C Interface to LLVM IR export ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/ExportLLVM.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

MlirLogicalResult mlirExportLLVMIR(MlirModule module,
                                   MlirStringCallback callback,
                                   void *userData) {
  mlir::ModuleOp moduleOp = unwrap(module);
  mlir::detail::CallbackOstream stream(callback, userData);

  mlir::registerBuiltinDialectTranslation(*moduleOp->getContext());
  mlir::registerLLVMDialectTranslation(*moduleOp->getContext());

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
  if (!llvmModule)
    return mlirLogicalResultFailure();

  llvmModule->print(stream, nullptr);
  return mlirLogicalResultSuccess();
}
