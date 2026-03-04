//===- FSM.cpp - C interface for the FSM dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/FSM.h"
#include "circt/Conversion/Passes.h"
#include "circt/Dialect/FSM/FSMDialect.h"
#include "circt/Dialect/FSM/FSMPasses.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt::fsm;

void registerFSMPasses() {
  registerPasses();
  circt::registerConvertFSMToSVPass();
  circt::registerConvertFSMToSMTPass();
  circt::registerConvertFSMToCorePass();
}
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FSM, fsm, FSMDialect)
