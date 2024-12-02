//===- FIRAnnotations.h - .fir file annotation interface --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains private APIs the parser uses to load annotations.
//
//===----------------------------------------------------------------------===//

#ifndef FIRANNOTATIONS_H
#define FIRANNOTATIONS_H

namespace circt {
namespace firrtl {

class PrintFOp;

/// Classifier for legacy verif intent captured in printf + when's.  Returns
/// true if the printf encodes verif intent, false otherwise.
bool isRecognizedPrintfEncodedVerif(PrintFOp printOp);

} // namespace firrtl
} // namespace circt

#endif // FIRANNOTATIONS_H
