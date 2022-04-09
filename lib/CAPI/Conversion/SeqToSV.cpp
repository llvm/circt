//===- SeqToSV.cpp - C Interface for the SeqToSV Conversion ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt-c/Conversion/SeqToSV.h"
#include "circt/Conversion/Passes.h"

#include "mlir/CAPI/Registration.h"

void registerConvertSeqToSVPass() { circt::registerConvertSeqToSVPass(); }
