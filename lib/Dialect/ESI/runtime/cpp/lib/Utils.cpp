//===- Utils.cpp - implementations ESI utility code -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/Utils.h"

#ifdef __GNUG__
#include <cstdlib>
#include <cxxabi.h>
#include <memory>

#endif

static constexpr char Table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "abcdefghijklmnopqrstuvwxyz"
                                "0123456789+/";

// Copied then modified slightly from llvm/lib/Support/Base64.cpp.
void esi::utils::encodeBase64(const void *dataIn, size_t size,
                              std::string &buffer) {
  const char *data = static_cast<const char *>(dataIn);
  buffer.resize(((size + 2) / 3) * 4);

  size_t i = 0, j = 0;
  for (size_t n = size / 3 * 3; i < n; i += 3, j += 4) {
    uint32_t x = ((unsigned char)data[i] << 16) |
                 ((unsigned char)data[i + 1] << 8) | (unsigned char)data[i + 2];
    buffer[j + 0] = Table[(x >> 18) & 63];
    buffer[j + 1] = Table[(x >> 12) & 63];
    buffer[j + 2] = Table[(x >> 6) & 63];
    buffer[j + 3] = Table[x & 63];
  }
  if (i + 1 == size) {
    uint32_t x = ((unsigned char)data[i] << 16);
    buffer[j + 0] = Table[(x >> 18) & 63];
    buffer[j + 1] = Table[(x >> 12) & 63];
    buffer[j + 2] = '=';
    buffer[j + 3] = '=';
  } else if (i + 2 == size) {
    uint32_t x =
        ((unsigned char)data[i] << 16) | ((unsigned char)data[i + 1] << 8);
    buffer[j + 0] = Table[(x >> 18) & 63];
    buffer[j + 1] = Table[(x >> 12) & 63];
    buffer[j + 2] = Table[(x >> 6) & 63];
    buffer[j + 3] = '=';
  }
}

std::string esi::utils::demangle(const std::type_info &ti) {
#ifdef __GNUG__
  int status = 0;
  std::unique_ptr<char, void (*)(void *)> res{
      abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status), std::free};
  return (status == 0) ? res.get() : ti.name();
#elif defined(_MSC_VER)
  return ti.name(); // MSVC already provides demangled names with typeid
#else
  return ti.name(); // Default: no demangling
#endif
}
