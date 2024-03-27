//===- Debug.cpp - Debug Utilities ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities related to generating run-time debug information.
//
//===----------------------------------------------------------------------===//

#include "circt/Support/Debug.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

llvm::raw_ostream &circt::debugHeader(llvm::StringRef str, int width) {
  auto &dbgStream = llvm::dbgs();
  dbgStream << "===- " << str << " ";
  width -= 6 + str.size();
  if (width > 3) {
    dbgStream << std::string(width - 3, '-');
    width = 3;
  }
  if (width > 0)
    dbgStream << std::string(width, '=');
  return dbgStream;
}

llvm::raw_ostream &circt::debugPassHeader(const mlir::Pass *pass, int width) {
  return debugHeader((llvm::Twine("Running ") + pass->getName()).str());
}

llvm::raw_ostream &circt::debugFooter(int width) {
  auto &dbgStream = llvm::dbgs();
  if (width > 3) {
    int startWidth = std::min(width - 3, 3);
    dbgStream << std::string(startWidth, '=');
    width -= startWidth;
  }
  if (width > 3) {
    dbgStream << std::string(width - 3, '-');
    width = 3;
  }
  if (width > 0)
    dbgStream << std::string(width, '=');
  return dbgStream;
}
