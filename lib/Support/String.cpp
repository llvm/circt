//===- String.cpp - String Utilities ----------------------------*- C++ -*-===//
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

#include "circt/Support/String.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

llvm::Optional<std::string> circt::unescape(StringRef str) {
  std::string result;
  result.reserve(str.size());
  for (size_t i = 0, e = str.size(); i != e;) {
    auto c = str[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    if (i >= e)
      return llvm::None;

    char c1 = str[i++];
    switch (c1) {
    case '"':
    case '\'':
    case '\\':
      result.push_back(c1);
      continue;
    case 'b':
      result.push_back('\b');
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    case 'f':
      result.push_back('\f');
      continue;
    case 'r':
      result.push_back('\r');
      continue;
    default:
      break;
    }

    if (i >= e)
      return llvm::None;

    auto c2 = str[i++];
    if (!llvm::isHexDigit(c1) || !llvm::isHexDigit(c2))
      return llvm::None;

    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }
  return result;
}

std::string circt::escape(StringRef str) {
  std::string result;
  llvm::raw_string_ostream os(result);

  for (unsigned char c : str) {
    switch (c) {
    case '\\':
      os << '\\' << '\\';
      break;
    case '\b':
      os << '\\' << 'b';
      break;
    case '\n':
      os << '\\' << 'n';
      break;
    case '\t':
      os << '\\' << 't';
      break;
    case '\f':
      os << '\\' << 'f';
      break;
    case '"':
      os << '\\' << '"';
      break;
    case '\'':
      os << '\\' << '\'';
      break;
    default:
      if (llvm::isPrint(c)) {
        os << c;
      } else {
        os << '\\';
        os << llvm::hexdigit((c >> 4) & 0xF);
        os << llvm::hexdigit((c >> 0) & 0xF);
      }
      break;
    }
  }
  return result;
}
