//===- AffineToHandshake.h --------------------------------------*- C++ -*-===//
//
// Copyright 2020 The CIRCT Authors.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// =============================================================================

#ifndef CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_
#define CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_

namespace circt {
namespace handshake {
void registerAffineToHandshakePasses();
}
} // namespace circt

#endif // CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_
