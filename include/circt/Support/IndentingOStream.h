//===- IndentingOStream.h - Indentable ostream ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements an indentable ostream.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_INDENTINGOSTREAM_H
#define CIRCT_SUPPORT_INDENTINGOSTREAM_H

#include "llvm/Support/Format.h"

namespace circt {
namespace support {

class indenting_ostream {
public:
  indenting_ostream(llvm::raw_ostream &os, unsigned amt = 2)
      : os(os), amt(amt) {}

  template <typename T>
  indenting_ostream &operator<<(T t) {
    os << t;
    return *this;
  }

  indenting_ostream &indent() {
    os.indent(currentIndent);
    return *this;
  }
  indenting_ostream &pad(size_t space) {
    os.indent(space);
    return *this;
  }
  void addIndent() { currentIndent += amt; }
  void reduceIndent() { currentIndent -= amt; }

  llvm::raw_ostream &getStream() { return os; }

private:
  llvm::raw_ostream &os;
  size_t currentIndent = 0;
  unsigned amt = 2;
};

} // namespace support
} // namespace circt

#endif // CIRCT_SUPPORT_INDENTINGOSTREAM_H
