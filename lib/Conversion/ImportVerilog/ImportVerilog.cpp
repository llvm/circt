//===- ImportVerilog.cpp - Slang Verilog frontend integration -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements bridging from the slang Verilog frontend to CIRCT dialects.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ImportVerilog.h"
#include "llvm/Support/raw_ostream.h"

#include "slang/util/Version.h"

std::string circt::getSlangVersion() {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  os << "slang version ";
  os << slang::VersionInfo::getMajor() << ".";
  os << slang::VersionInfo::getMinor() << ".";
  os << slang::VersionInfo::getPatch() << "+";
  os << slang::VersionInfo::getHash();
  return buffer;
}
