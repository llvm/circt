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

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <new>

void *esi::utils::alignedAlloc(size_t size, size_t alignment) {
  if (size == 0 || (alignment && !std::has_single_bit(alignment)))
    return nullptr;
  if (alignment == 0)
    return std::malloc(size);
  alignment = std::max(alignment, alignof(std::max_align_t));
  // Aligned allocation implementations may round the requested size up.
  if (size > std::numeric_limits<size_t>::max() - (alignment - 1))
    return nullptr;
  return ::operator new(size, std::align_val_t(alignment), std::nothrow);
}

void esi::utils::alignedFree(void *ptr, size_t alignment) {
  if (!ptr)
    return;
  if (alignment == 0)
    std::free(ptr);
  else
    ::operator delete(
        ptr, std::align_val_t(std::max(alignment, alignof(std::max_align_t))));
}

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
