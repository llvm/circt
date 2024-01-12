//===- Ports.cpp - ESI communication channels  -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Ports.h"

#include <stdexcept>

using namespace std;
using namespace esi;

BundlePort::BundlePort(AppID id, map<string, ChannelPort &> channels)
    : id(id), channels(channels) {}

WriteChannelPort &BundlePort::getRawWrite(const string &name) const {
  auto f = channels.find(name);
  if (f == channels.end())
    throw runtime_error("Channel '" + name + "' not found");
  auto *write = dynamic_cast<WriteChannelPort *>(&f->second);
  if (!write)
    throw runtime_error("Channel '" + name + "' is not a write channel");
  return *write;
}

ReadChannelPort &BundlePort::getRawRead(const string &name) const {
  auto f = channels.find(name);
  if (f == channels.end())
    throw runtime_error("Channel '" + name + "' not found");
  auto *read = dynamic_cast<ReadChannelPort *>(&f->second);
  if (!read)
    throw runtime_error("Channel '" + name + "' is not a read channel");
  return *read;
}
