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
#include <filesystem>
#include <map>
#include <stdexcept>

#include <iostream>

#ifdef __linux__
#include <dlfcn.h>
#include <linux/limits.h>
#include <unistd.h>
#elif _WIN32
// TODO: this.
#endif

using namespace esi;
using namespace esi::services;

namespace esi {
AcceleratorConnection::AcceleratorConnection(Context &ctxt)
    : ctxt(ctxt), serviceThread(std::make_unique<AcceleratorServiceThread>()) {}

services::Service *AcceleratorConnection::getService(Service::Type svcType,
                                                     AppIDPath id,
                                                     std::string implName,
                                                     ServiceImplDetails details,
                                                     HWClientDetails clients) {
  std::unique_ptr<Service> &cacheEntry = serviceCache[make_tuple(&svcType, id)];
  if (cacheEntry == nullptr) {
    Service *svc = createService(svcType, id, implName, details, clients);
    if (!svc)
      svc = ServiceRegistry::createService(this, svcType, id, implName, details,
                                           clients);
    if (!svc)
      return nullptr;
    cacheEntry = std::unique_ptr<Service>(svc);
  }
  return cacheEntry.get();
}

/// Get the path to the currently running executable.
static std::filesystem::path getExePath() {
#ifdef __linux__
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  if (count == -1)
    throw std::runtime_error("Could not get executable path");
  return std::filesystem::path(std::string(result, count));
#elif _WIN32
  // TODO: this.
#else
#eror "Unsupported platform"
#endif
}

/// Get the path to the currently running shared library.
static std::filesystem::path getLibPath() {
#ifdef __linux__
  Dl_info dl_info;
  dladdr((void *)getLibPath, &dl_info);
  return std::filesystem::path(std::string(dl_info.dli_fname));
#elif _WIN32
  // TODO: this.
#else
#eror "Unsupported platform"
#endif
}

/// Load a backend plugin dynamically. Plugins are expected to be named
/// lib<BackendName>Backend.so and located in one of 1) CWD, 2) in the same
/// directory as the application, or 3) in the same directory as this library.
static void loadBackend(std::string backend) {
  backend[0] = toupper(backend[0]);

  // Get the file name we are looking for.
#ifdef __linux__
  std::string backendFileName = "lib" + backend + "Backend.so";
#elif _WIN32
  // TODO: this.
  return;
#else
#eror "Unsupported platform"
#endif

  // Look for library using the C++ std API.
  // TODO: once the runtime has a logging framework, log the paths we are
  // trying.

  // First, try the current directory.
  std::filesystem::path backendPath = backendFileName;
  if (!std::filesystem::exists(backendPath)) {
    // Next, try the directory of the executable.
    backendPath = getExePath().parent_path().append(backendFileName);
    if (!std::filesystem::exists(backendPath)) {
      // Finally, try
      backendPath = getLibPath().parent_path().append(backendFileName);
      if (!std::filesystem::exists(backendPath))
        throw std::runtime_error("Backend library not found");
    }
  }

  // Attempt to load it.
#ifdef __linux__
  void *handle = dlopen(backendPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle)
    throw std::runtime_error("While attempting to load backend plugin: " +
                             std::string(dlerror()));
#elif _WIN32
  // TODO: this.
#else
#eror "Unsupported platform"
#endif
}

namespace registry {
namespace internal {

static std::map<std::string, BackendCreate> backendRegistry;
void registerBackend(std::string name, BackendCreate create) {
  if (backendRegistry.count(name))
    throw std::runtime_error("Backend already exists in registry");
  backendRegistry[name] = create;
}
} // namespace internal

std::unique_ptr<AcceleratorConnection>
connect(Context &ctxt, std::string backend, std::string connection) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end()) {
    // If it's not already found in the registry, try to load it dynamically.
    loadBackend(backend);
    f = internal::backendRegistry.find(backend);
    if (f == internal::backendRegistry.end())
      throw std::runtime_error("Backend '" + backend + "' not found");
  }
  return f->second(ctxt, connection);
}

} // namespace registry

struct AcceleratorServiceThread::Impl {
  Impl() {}
  void start() { me = std::thread(&Impl::loop, this); }
  void stop() {
    shutdown = true;
    me.join();
  }
  /// When there's data on any of the listenPorts, call the callback. This
  /// method can be called from any thread.
  void
  addListener(std::initializer_list<ReadChannelPort *> listenPorts,
              std::function<void(ReadChannelPort *, MessageData)> callback);

private:
  void loop();
  volatile bool shutdown = false;
  std::thread me;

  // Protect the listeners std::map.
  std::mutex listenerMutex;
  // Map of read ports to callbacks.
  std::map<ReadChannelPort *,
           std::pair<std::function<void(ReadChannelPort *, MessageData)>,
                     std::future<MessageData>>>
      listeners;
};

void AcceleratorServiceThread::Impl::loop() {
  // These two variables should logically be in the loop, but this avoids
  // reconstructing them on each iteration.
  std::vector<std::tuple<ReadChannelPort *,
                         std::function<void(ReadChannelPort *, MessageData)>,
                         MessageData>>
      portUnlockWorkList;
  MessageData data;

  while (!shutdown) {
    // Ideally we'd have some wake notification here, but this sufficies for
    // now.
    // TODO: investigate better ways to do this.
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    // Check and gather data from all the read ports we are monitoring. Put the
    // callbacks to be called later so we can release the lock.
    {
      std::lock_guard<std::mutex> g(listenerMutex);
      for (auto &[channel, cbfPair] : listeners) {
        assert(channel && "Null channel in listener list");
        std::future<MessageData> &f = cbfPair.second;
        if (f.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
          portUnlockWorkList.emplace_back(channel, cbfPair.first, f.get());
          f = channel->readAsync();
        }
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
      throw std::runtime_error("Port already has a listener");
    listeners[port] = std::make_pair(callback, port->readAsync());
  }
}

} // namespace esi

AcceleratorServiceThread::AcceleratorServiceThread()
    : impl(std::make_unique<Impl>()) {
  impl->start();
}
AcceleratorServiceThread::~AcceleratorServiceThread() { stop(); }

void AcceleratorServiceThread::stop() {
  if (impl) {
    impl->stop();
    impl.reset();
  }
}

// When there's data on any of the listenPorts, call the callback. This is
// kinda silly now that we have callback port support, especially given the
// polling loop. Keep the functionality for now.
void AcceleratorServiceThread::addListener(
    std::initializer_list<ReadChannelPort *> listenPorts,
    std::function<void(ReadChannelPort *, MessageData)> callback) {
  assert(impl && "Service thread not running");
  impl->addListener(listenPorts, callback);
}

void AcceleratorConnection::disconnect() {
  if (serviceThread) {
    serviceThread->stop();
    serviceThread.reset();
  }
}
