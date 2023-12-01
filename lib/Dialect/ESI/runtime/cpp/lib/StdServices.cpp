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

#include "zlib.h"

#include <cassert>
#include <stdexcept>

using namespace std;

using namespace esi;
using namespace esi::services;

string SysInfo::getServiceSymbol() const { return "__builtin_SysInfo"; }

// Allocate 10MB for the uncompressed manifest. This should be plenty.
constexpr uint32_t MAX_MANIFEST_SIZE = 10 << 20;
/// Get the compressed manifest, uncompress, and return it.
string SysInfo::getJsonManifest() const {
  vector<uint8_t> compressed = getCompressedManifest();
  vector<Bytef> dst(MAX_MANIFEST_SIZE);
  uLongf dstSize = MAX_MANIFEST_SIZE;
  int rc =
      uncompress(dst.data(), &dstSize, compressed.data(), compressed.size());
  if (rc != Z_OK)
    throw runtime_error("zlib uncompress failed with rc=" + to_string(rc));
  return string(reinterpret_cast<char *>(dst.data()), dstSize);
}

string MMIO::getServiceSymbol() const { return "__builtin_MMIO"; }

MMIOSysInfo::MMIOSysInfo(const MMIO *mmio) : mmio(mmio) {}

uint32_t MMIOSysInfo::getEsiVersion() const {
  uint32_t hi = mmio->read(MagicNumOffset);
  uint32_t lo = mmio->read(MagicNumOffset + 4);
  if (hi != MagicNumberHi || lo != MagicNumberLo)
    throw runtime_error("ESI magic number not found");
  return mmio->read(VersionNumberOffset);
}

vector<uint8_t> MMIOSysInfo::getCompressedManifest() const {
  assert(false && "Not implemented");
  throw runtime_error("Not implemented");
}
