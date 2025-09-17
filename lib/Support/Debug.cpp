//===----------------------------------------------------------------------===//
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

llvm::raw_ostream &circt::debugHeader(const llvm::Twine &str, unsigned width) {
  auto &os = llvm::dbgs();
  auto start = os.tell();
  os << "===- " << str << " ";
  auto endAct = os.tell() + 4; // 4 for "-==="
  auto endExp = start + width;
  while (endAct < endExp) {
    os << '-';
    ++endAct;
  }
  os << "-===";
  return os;
}

llvm::raw_ostream &circt::debugPassHeader(const mlir::Pass *pass,
                                          unsigned width) {
  return debugHeader(llvm::Twine("Running ") + pass->getName());
}

llvm::raw_ostream &circt::debugFooter(unsigned width) {
  auto &os = llvm::dbgs();
  auto start = os.tell();
  os << "===";
  auto endAct = os.tell() + 3; // 3 for trailing "==="
  auto endExp = start + width;
  while (endAct < endExp) {
    os << '-';
    ++endAct;
  }
  os << "===";
  return os;
}
