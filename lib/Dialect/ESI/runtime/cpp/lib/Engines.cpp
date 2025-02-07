//===- Engines.cpp --------------------------------------------------------===//
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

#include "esi/Engines.h"

using namespace esi;

ChannelPort &Engine::requestPort(AppIDPath idPath,
                                 const std::string &channelName,
                                 BundleType::Direction dir, const Type *type) {
  auto portIter = ownedPorts.find(std::make_pair(idPath, channelName));
  if (portIter != ownedPorts.end())
    return *portIter->second;
  std::unique_ptr<ChannelPort> port =
      createPort(idPath, channelName, dir, type);
  ChannelPort &ret = *port;
  ownedPorts.emplace(std::make_pair(idPath, channelName), std::move(port));
  return ret;
}

PortMap BundleEngineMap::requestPorts(const AppIDPath &idPath,
                                      const BundleType *bundleType) const {
  PortMap ports;
  for (auto [channelName, dir, type] : bundleType->getChannels()) {
    auto engineIter = bundleEngineMap.find(channelName);
    if (engineIter == bundleEngineMap.end())
      continue;

    ports.emplace(channelName, engineIter->second->requestPort(
                                   idPath, channelName, dir, type));
  }
  return ports;
}

void BundleEngineMap::setEngine(const std::string &channelName,
                                Engine *engine) {
  auto [it, inserted] = bundleEngineMap.try_emplace(channelName, engine);
  if (!inserted)
    throw std::runtime_error("Channel already exists in engine map");
}

namespace {
std::map<std::string, registry::internal::EngineCreate> engineRegistry;
}

std::unique_ptr<Engine>
registry::createEngine(AcceleratorConnection &conn,
                       const std::string &dmaEngineName, AppIDPath idPath,
                       const ServiceImplDetails &details,
                       const HWClientDetails &clients) {
  auto it = engineRegistry.find(dmaEngineName);
  if (it == engineRegistry.end())
    throw std::runtime_error("Unknown engine: " + dmaEngineName);
  return it->second(conn, idPath, details, clients);
}

void registry::internal::registerEngine(const std::string &name,
                                        EngineCreate create) {
  auto tried = engineRegistry.try_emplace(name, create);
  if (!tried.second)
    throw std::runtime_error("Engine already exists in registry");
}
