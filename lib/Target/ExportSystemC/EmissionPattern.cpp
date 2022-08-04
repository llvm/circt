//===- EmissionPattern.cpp - Emission Pattern Base and Utility impl -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the emission pattern base and utility classes.
//
//===----------------------------------------------------------------------===//

#include "EmissionPattern.h"
#include "EmissionPrinter.h"

using namespace circt;
using namespace circt::ExportSystemC;

EmissionResult::EmissionResult() : isFailure(true) {}

EmissionResult::EmissionResult(mlir::StringRef expression,
                               Precedence precedence)
    : isFailure(false), precedence(precedence), expression(expression) {}

EmissionPatternSet::EmissionPatternSet(
    std::vector<std::unique_ptr<EmissionPattern>> &patterns)
    : patterns(patterns) {}
