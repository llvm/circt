//===- PrettyPrinterHelpers.cpp - Pretty printing helpers -----------------===//
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

#include "circt/Support/PrettyPrinterHelpers.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Convenience builders.
//===----------------------------------------------------------------------===//

void TokenStringSaver::clear() { alloc.Reset(); }

namespace detail {

/// Add multiple non-breaking spaces as a single token.
void emitNBSP(unsigned n, llvm::function_ref<void(Token)> add) {
  static const char spaces[] = {
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
      ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
  };
  auto size = std::size(spaces);
  if (n < size) {
    if (n != 0)
      add(StringToken({spaces, n}));
    return;
  }
  while (n) {
    auto chunk = std::min<uint32_t>(n, size - 1);
    add(StringToken({spaces, chunk}));
    n -= chunk;
  }
}
} // end namespace detail

} // end namespace pretty
} // end namespace circt
