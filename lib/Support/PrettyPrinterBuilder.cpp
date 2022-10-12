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

//===----------------------------------------------------------------------===//
// Streaming support.
//===----------------------------------------------------------------------===//

PPStream &PPStream::writeQuotedEscaped(StringRef str, bool useHexEscapes,
                                       StringRef left, StringRef right) {
  SmallString<64> ss;
  {
    llvm::raw_svector_ostream os(ss);
    os << left;
    os.write_escaped(str, useHexEscapes);
    os << right;
  }
  return *this << ss;
}

} // end namespace pretty
} // end namespace circt
