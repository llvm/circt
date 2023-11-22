//===- Design.cpp - Implementation of dynamic API -------------------------===//
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

#include "esi/Design.h"
#include "esi/Accelerator.h"

using namespace std;
using namespace esi;

BundlePort::BundlePort(AppID id, map<string, ChannelPort &> channels)
    : _id(id), _channels(channels) {}

WriteChannelPort &BundlePort::getRawWrite(const string &name) const {
  auto f = _channels.find(name);
  if (f == _channels.end())
    throw runtime_error("Channel '" + name + "' not found");
  auto *write = dynamic_cast<WriteChannelPort *>(&f->second);
  if (!write)
    throw runtime_error("Channel '" + name + "' is not a write channel");
  return *write;
}

ReadChannelPort &BundlePort::getRawRead(const string &name) const {
  auto f = _channels.find(name);
  if (f == _channels.end())
    throw runtime_error("Channel '" + name + "' not found");
  auto *read = dynamic_cast<ReadChannelPort *>(&f->second);
  if (!read)
    throw runtime_error("Channel '" + name + "' is not a read channel");
  return *read;
}

/// Build an index of children by AppID.
static map<AppID, Instance *>
buildIndex(const vector<unique_ptr<Instance>> &insts) {
  map<AppID, Instance *> index;
  for (auto &item : insts)
    index[item->getID()] = item.get();
  return index;
}

/// Build an index of ports by AppID.
static map<AppID, const BundlePort &>
buildIndex(const vector<BundlePort> &ports) {
  map<AppID, const BundlePort &> index;
  for (auto &item : ports)
    index.emplace(item.getID(), item);
  return index;
}

Design::Design(std::optional<ModuleInfo> info,
               std::vector<std::unique_ptr<Instance>> children,
               std::vector<services::Service *> services,
               std::vector<BundlePort> ports)
    : info(info), children(std::move(children)),
      childIndex(buildIndex(this->children)), services(services), ports(ports),
      portIndex(buildIndex(this->ports)) {}
