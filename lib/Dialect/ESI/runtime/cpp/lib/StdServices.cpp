//===- StdServices.cpp - implementations of std services ------------------===//
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

#include "esi/StdServices.h"

#include <stdexcept>

using namespace esi;
using namespace esi::services;

MMIOSysInfo::MMIOSysInfo(const MMIO *mmio) : mmio(mmio) {}

uint32_t MMIOSysInfo::esiVersion() const {
  uint32_t hi = mmio->read(MagicNumOffset);
  uint32_t lo = mmio->read(MagicNumOffset + 4);
  if (hi != MagicNumberHi || lo != MagicNumberLo)
    throw std::runtime_error("ESI magic number not found");
  return mmio->read(VersionNumberOffset);
}

std::string MMIOSysInfo::rawJsonManifest() const { return ""; }
