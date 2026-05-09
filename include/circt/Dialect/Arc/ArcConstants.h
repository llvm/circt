//===- ArcConstants.h - Declare Arc dialect constants ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_ARCCONSTANTS_H
#define CIRCT_DIALECT_ARC_ARCCONSTANTS_H

namespace circt {
namespace arc {

// Offsets for model state allocations.
inline constexpr unsigned kTimeOffset = 0;
inline constexpr unsigned kTerminateFlagOffset = 8;
inline constexpr unsigned kNextWakeupOffset = 16;
inline constexpr unsigned kStateOffset = 24;

} // namespace arc

} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCCONSTANTS_H
