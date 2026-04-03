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

// Offset for model state allocations. The first 16 bytes of model storage are
// reserved for the model header. The first 8 bytes contain the current
// simulation time. The next 8 bytes are reserved for the termination flag used
// by `SimTerminateOp`. The actual model state starts at offset 16.
inline constexpr unsigned kTimeOffset = 0;
inline constexpr unsigned kTerminateFlagOffset = 8;
inline constexpr unsigned kStateOffset = 16;

} // namespace arc

} // namespace circt

#endif // CIRCT_DIALECT_ARC_ARCCONSTANTS_H
