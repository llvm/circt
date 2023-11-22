//===- Utils.h - ESI runtime utility code -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_UTILS_H
#define ESI_UTILS_H

#include <cstdint>
#include <string>

namespace esi {
namespace utils {
// Very basic base64 encoding.
void encodeBase64(const void *data, size_t size, std::string &out);
} // namespace utils
} // namespace esi

#endif // ESI_UTILS_H
