//===-- Moore.h - C API for Moore dialect ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Dialect/Moore.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"

using namespace circt;
using namespace circt::moore;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Moore, moore, MooreDialect)
