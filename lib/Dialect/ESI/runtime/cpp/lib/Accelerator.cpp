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
#include <windows.h>
#endif

using namespace esi;
using namespace esi::services;

namespace esi {
AcceleratorConnection::AcceleratorConnection(Context &ctxt)
    : ctxt(ctxt), serviceThread(nullptr) {}
AcceleratorConnection::~AcceleratorConnection() { disconnect(); }

AcceleratorServiceThread *AcceleratorConnection::getServiceThread() {
  if (!serviceThread)
    serviceThread = std::make_unique<AcceleratorServiceThread>();
  return serviceThread.get();
}
void AcceleratorConnection::createEngine(const std::string &engineTypeName,
                                         AppIDPath idPath,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  std::unique_ptr<Engine> engine = ::esi::registry::createEngine(
      *this, engineTypeName, idPath, details, clients);
  registerEngine(idPath, std::move(engine), clients);
}

void AcceleratorConnection::registerEngine(AppIDPath idPath,
                                           std::unique_ptr<Engine> engine,
                                           const HWClientDetails &clients) {
  assert(engine);
  auto [engineIter, _] = ownedEngines.emplace(idPath, std::move(engine));

  // Engine is now owned by the accelerator connection, so the std::unique_ptr
  // is no longer valid. Resolve a new one from the map iter.
  Engine *enginePtr = engineIter->second.get();
  // Compute our parents idPath path.
  AppIDPath prefix = std::move(idPath);
  if (prefix.size() > 0)
    prefix.pop_back();

  for (const auto &client : clients) {
    AppIDPath fullClientPath = prefix + client.relPath;
    for (const auto &channel : client.channelAssignments)
      clientEngines[fullClientPath].setEngine(channel.first, enginePtr);
  }
}

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

Accelerator *
AcceleratorConnection::takeOwnership(std::unique_ptr<Accelerator> acc) {
  if (ownedAccelerator)
    throw std::runtime_error(
        "AcceleratorConnection already owns an accelerator");
  ownedAccelerator = std::move(acc);
  return ownedAccelerator.get();
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
  char buffer[MAX_PATH];
  DWORD length = GetModuleFileNameA(NULL, buffer, MAX_PATH);
  if (length == 0)
    throw std::runtime_error("Could not get executable path");
  return std::filesystem::path(std::string(buffer, length));
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
  HMODULE hModule = NULL;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&getLibPath), &hModule)) {
    // Handle error
    return std::filesystem::path();
  }

  char buffer[MAX_PATH];
  DWORD length = GetModuleFileNameA(hModule, buffer, MAX_PATH);
  if (length == 0)
    throw std::runtime_error("Could not get library path");

  return std::filesystem::path(std::string(buffer, length));
#else
#eror "Unsupported platform"
#endif
}

/// Load a backend plugin dynamically. Plugins are expected to be named
/// lib<BackendName>Backend.so and located in one of 1) CWD, 2) in the same
/// directory as the application, or 3) in the same directory as this library.
static void loadBackend(Context &ctxt, std::string backend) {
  Logger &logger = ctxt.getLogger();
  backend[0] = toupper(backend[0]);

  // Get the file name we are looking for.
#ifdef __linux__
  std::string backendFileName = "lib" + backend + "Backend.so";
#elif _WIN32
  std::string backendFileName = backend + "Backend.dll";
#else
#eror "Unsupported platform"
#endif

  // Look for library using the C++ std API.

  // First, try the current directory.
  std::filesystem::path backendPath = backendFileName;
  std::string backendPathStr;
  logger.debug("CONNECT",
               "trying to find backend plugin: " + backendPath.string());
  if (!std::filesystem::exists(backendPath)) {
    // Next, try the directory of the executable.
    backendPath = getExePath().parent_path().append(backendFileName);
    logger.debug("CONNECT",
                 "trying to find backend plugin: " + backendPath.string());
    if (!std::filesystem::exists(backendPath)) {
      // Finally, try the directory of the library.
      backendPath = getLibPath().parent_path().append(backendFileName);
      logger.debug("CONNECT",
                   "trying to find backend plugin: " + backendPath.string());
      if (!std::filesystem::exists(backendPath)) {
        // If all else fails, just try the name.
        backendPathStr = backendFileName;
        logger.debug("CONNECT",
                     "trying to find backend plugin: " + backendPathStr);
      }
    }
  }
  // If the path was found, convert it to a string.
  if (backendPathStr.empty())
    backendPathStr = backendPath.string();
  else
    // Otherwise, signal that the path wasn't found by clearing the path and
    // just use the name. (This is only used on Windows to add the same
    // directory as the backend DLL to the DLL search path.)
    backendPath.clear();

    // Attempt to load it.
#ifdef __linux__
  void *handle = dlopen(backendPathStr.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    std::string error(dlerror());
    logger.error("CONNECT",
                 "while attempting to load backend plugin: " + error);
    throw std::runtime_error("While attempting to load backend plugin: " +
                             error);
  }
#elif _WIN32
  // Set the DLL directory to the same directory as the backend DLL in case it
  // has transitive dependencies.
  if (backendPath != std::filesystem::path()) {
    std::filesystem::path backendPathParent = backendPath.parent_path();
    if (backendPathParent.empty())
      backendPathParent = std::filesystem::current_path();
    logger.debug("CONNECT", "setting DLL search directory to: " +
                                backendPathParent.string());
    if (SetDllDirectoryA(backendPathParent.string().c_str()) == 0)
      throw std::runtime_error("While setting DLL directory: " +
                               std::to_string(GetLastError()));
  }

  // Load the backend plugin.
  HMODULE handle = LoadLibraryA(backendPathStr.c_str());
  if (!handle) {
    DWORD error = GetLastError();
    if (error == ERROR_MOD_NOT_FOUND) {
      logger.error("CONNECT", "while attempting to load backend plugin: " +
                                  backendPathStr + " not found");
      throw std::runtime_error("While attempting to load backend plugin: " +
                               backendPathStr + " not found");
    }
    logger.error("CONNECT", "while attempting to load backend plugin: " +
                                std::to_string(error));
    throw std::runtime_error("While attempting to load backend plugin: " +
                             std::to_string(error));
  }
#else
#eror "Unsupported platform"
#endif
  logger.info("CONNECT", "loaded backend plugin: " + backendPathStr);
}

namespace registry {
namespace internal {

class BackendRegistry {
public:
  static std::map<std::string, BackendCreate> &get() {
    static BackendRegistry instance;
    return instance.backendRegistry;
  }

private:
  std::map<std::string, BackendCreate> backendRegistry;
};

void registerBackend(const std::string &name, BackendCreate create) {
  auto &registry = BackendRegistry::get();
  if (registry.count(name))
    throw std::runtime_error("Backend already exists in registry");
  registry[name] = create;
}
} // namespace internal

std::unique_ptr<AcceleratorConnection> connect(Context &ctxt,
                                               const std::string &backend,
                                               const std::string &connection) {
  auto &registry = internal::BackendRegistry::get();
  auto f = registry.find(backend);
  if (f == registry.end()) {
    // If it's not already found in the registry, try to load it dynamically.
    loadBackend(ctxt, backend);
    f = registry.find(backend);
    if (f == registry.end()) {
      esi::ServiceImplDetails details;
      details["backend"] = backend;
      std::string loaded_backends;
      bool first = true;
      for (const auto &b : registry) {
        if (!first)
          loaded_backends += ", ";
        loaded_backends += b.first;
        first = false;
      }
      details["loaded_backends"] = loaded_backends;
      ctxt.getLogger().error("CONNECT", "backend '" + backend + "' not found",
                             &details);
      throw std::runtime_error("Backend '" + backend + "' not found");
    }
  }
  ctxt.getLogger().info("CONNECT", "connecting to backend " + backend +
                                       " via '" + connection + "'");
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

  void addTask(std::function<void(void)> task) {
    std::lock_guard<std::mutex> g(m);
    taskList.push_back(task);
  }

private:
  void loop();
  volatile bool shutdown = false;
  std::thread me;

  // Protect the shared data structures.
  std::mutex m;

  // Map of read ports to callbacks.
  std::map<ReadChannelPort *,
           std::pair<std::function<void(ReadChannelPort *, MessageData)>,
                     std::future<MessageData>>>
      listeners;

  /// Tasks which should be called on every loop iteration.
  std::vector<std::function<void(void)>> taskList;
};

void AcceleratorServiceThread::Impl::loop() {
  // These two variables should logically be in the loop, but this avoids
  // reconstructing them on each iteration.
  std::vector<std::tuple<ReadChannelPort *,
                         std::function<void(ReadChannelPort *, MessageData)>,
                         MessageData>>
      portUnlockWorkList;
  std::vector<std::function<void(void)>> taskListCopy;
  MessageData data;

  while (!shutdown) {
    // Ideally we'd have some wake notification here, but this sufficies for
    // now.
    // TODO: investigate better ways to do this. For now, just play nice with
    // the other processes but don't waste time in between polling intervals.
    std::this_thread::yield();

    // Check and gather data from all the read ports we are monitoring. Put the
    // callbacks to be called later so we can release the lock.
    {
      std::lock_guard<std::mutex> g(m);
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

    // Call any tasks that have been added. Copy it first so we can release the
    // lock ASAP.
    {
      std::lock_guard<std::mutex> g(m);
      taskListCopy = taskList;
    }
    for (auto &task : taskListCopy)
      task();
  }
}

void AcceleratorServiceThread::Impl::addListener(
    std::initializer_list<ReadChannelPort *> listenPorts,
    std::function<void(ReadChannelPort *, MessageData)> callback) {
  std::lock_guard<std::mutex> g(m);
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

void AcceleratorServiceThread::addPoll(HWModule &module) {
  assert(impl && "Service thread not running");
  impl->addTask([&module]() { module.poll(); });
}

void AcceleratorConnection::disconnect() {
  if (serviceThread) {
    serviceThread->stop();
    serviceThread.reset();
  }
}
