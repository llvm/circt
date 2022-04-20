//===- String.h - String Utilities ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for working with strings.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_STRING_H
#define CIRCT_SUPPORT_STRING_H

#include "circt/Support/LLVM.h"

namespace circt {

/// Converts escape sequences to characters.
llvm::Optional<std::string> unescape(StringRef str);

/// Escapes special characters in a string.
std::string escape(StringRef str);

} // namespace circt

#endif // CIRCT_SUPPORT_STRING_H
