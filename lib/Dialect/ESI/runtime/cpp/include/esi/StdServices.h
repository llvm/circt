//===- StdServices.h - ESI standard services C++ API ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The APIs in this backend are all optionally implemented. The lower level
// ones, however, are strongly recommended. 'Services' here refers to ESI
// services. These are standard APIs into the standard ESI services.
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_RUNTIME_STDSERVICES_H
#define ESI_RUNTIME_STDSERVICES_H

#include <cstdint>

namespace esi {
namespace services {

class MMIO {
public:
  uint64_t readReg(uint32_t addr);
  void writeReg(uint32_t addr, uint64_t data);
};

} // namespace services
} // namespace esi

#endif // ESI_RUNTIME_STDSERVICES_H
