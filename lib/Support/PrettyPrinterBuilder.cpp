//===- PrettyPrinterBuilder.cpp - Pretty printing builder -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for using PrettyPrinter.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/PrettyPrinterBuilder.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Convenience builders.
//===----------------------------------------------------------------------===//

void PPBuilderStringSaver::clear() { alloc.Reset(); }

} // end namespace pretty
} // end namespace circt
