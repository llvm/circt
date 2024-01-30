//===- ImportVerilog.h - Slang Verilog frontend integration ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interface to the Verilog frontend.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_IMPORTVERILOG_H
#define CIRCT_CONVERSION_IMPORTVERILOG_H

#include <string>

namespace circt {

/// Return a human-readable string describing the slang frontend version linked
/// into CIRCT.
std::string getSlangVersion();

} // namespace circt

#endif // CIRCT_CONVERSION_IMPORTVERILOG_H
