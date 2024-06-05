//===- Accelerator.cpp - ESI accelerator system API -----------------------===//
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

#include "esi/Accelerator.h"

#include <cassert>
#include <map>
#include <stdexcept>

using namespace std;

using namespace esi;
using namespace esi::services;

namespace esi {
AcceleratorConnection::AcceleratorConnection(Context &ctxt,
                                             std::ostream *debugLog)
    : ctxt(ctxt), debugLogFile(debugLog),
      serviceThread(make_unique<AcceleratorServiceThread>()) {}

void AcceleratorConnection::debugLogInternal(const std::string &action,
                                             const std::string &id,
                                             std::string msg) const {
  *debugLogFile << '[' << action << ' ' << id << "] " << msg << "\n";
  debugLogFile->flush();
}

services::Service *AcceleratorConnection::getService(Service::Type svcType,
                                                     AppIDPath id,
                                                     std::string implName,
                                                     ServiceImplDetails details,
                                                     HWClientDetails clients) {
  unique_ptr<Service> &cacheEntry = serviceCache[make_tuple(&svcType, id)];
  if (cacheEntry == nullptr) {
    Service *svc = createService(svcType, id, implName, details, clients);
    if (!svc)
      svc = ServiceRegistry::createService(this, svcType, id, implName, details,
                                           clients);
    if (!svc)
      return nullptr;
    cacheEntry = unique_ptr<Service>(svc);
  }
  return cacheEntry.get();
}

namespace registry {
namespace internal {

static map<string, BackendCreate> backendRegistry;
void registerBackend(string name, BackendCreate create) {
  if (backendRegistry.count(name))
    throw runtime_error("Backend already exists in registry");
  backendRegistry[name] = create;
}
} // namespace internal

unique_ptr<AcceleratorConnection> connect(Context &ctxt, string backend,
                                          string connection,
                                          std::ostream *debugLog) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end())
    throw runtime_error("Backend not found");
  auto conn = f->second(ctxt, connection);
  conn->setDebugLog(debugLog);
  return conn;
}

} // namespace registry

struct AcceleratorServiceThread::Impl {
  Impl() {}
  void start() { me = std::thread(&Impl::loop, this); }
  void stop() {
    shutdown = true;
    me.join();
  }
  /// When there's data on any of the listenPorts, call the callback. This can
  /// be called from a separate thread.
  void
  addListener(std::initializer_list<ReadChannelPort *> listenPorts,
              std::function<void(ReadChannelPort *, MessageData)> callback);

private:
  void loop();
  volatile bool shutdown = false;
  std::thread me;
  std::mutex listenerMutex;
  std::map<ReadChannelPort *,
           std::function<void(ReadChannelPort *, MessageData)>>
      listeners;
};

void AcceleratorServiceThread::Impl::loop() {
  // These two logically should be in the loop, but this avoids reconstructing
  // on each iteration.
  std::vector<std::tuple<ReadChannelPort *,
                         std::function<void(ReadChannelPort *, MessageData)>,
                         MessageData>>
      portUnlockWorkList;
  MessageData data;

  while (!shutdown) {
    // Ideally we'd have some wake notification here, but this sufficies for
    // now.
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    // Check and gather data from all the read ports we are monitoring. Put the
    // callbacks to be called later so we can release the lock.
    {
      std::lock_guard<std::mutex> g(listenerMutex);
      for (auto &[channel, cb] : listeners) {
        assert(channel && "Null channel in listener list");
        if (channel->read(data))
          portUnlockWorkList.emplace_back(channel, cb, std::move(data));
      }
    }

    // Call the callbacks outside the lock.
    for (auto [channel, cb, data] : portUnlockWorkList)
      cb(channel, std::move(data));

    // Clear the worklist for the next iteration.
    portUnlockWorkList.clear();
  }
}

void AcceleratorServiceThread::Impl::addListener(
    std::initializer_list<ReadChannelPort *> listenPorts,
    std::function<void(ReadChannelPort *, MessageData)> callback) {
  std::lock_guard<std::mutex> g(listenerMutex);
  for (auto port : listenPorts) {
    if (listeners.count(port))
      throw runtime_error("Port already has a listener");
    listeners[port] = callback;
  }
}

AcceleratorServiceThread::AcceleratorServiceThread()
    : impl(std::make_unique<Impl>()) {
  impl->start();
}
AcceleratorServiceThread::~AcceleratorServiceThread() { impl->stop(); }

/// When there's data on any of the listenPorts, call the callback.
void AcceleratorServiceThread::addListener(
    std::initializer_list<ReadChannelPort *> listenPorts,
    std::function<void(ReadChannelPort *, MessageData)> callback) {
  impl->addListener(listenPorts, callback);
}

} // namespace esi
