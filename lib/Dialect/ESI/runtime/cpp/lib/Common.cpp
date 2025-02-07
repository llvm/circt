//===- Common.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Common.h"

#include <iostream>
#include <sstream>

using namespace esi;

std::string MessageData::toHex() const {
  std::ostringstream ss;
  ss << std::hex;
  for (size_t i = 0, e = data.size(); i != e; ++i) {
    // Add spaces every 8 bytes.
    if (i % 8 == 0 && i != 0)
      ss << ' ';
    // Add an extra space every 64 bytes.
    if (i % 64 == 0 && i != 0)
      ss << ' ';
    ss << static_cast<unsigned>(data[i]);
  }
  return ss.str();
}

std::string esi::toHex(uint64_t val) {
  std::ostringstream ss;
  ss << std::hex << val;
  return ss.str();
}
