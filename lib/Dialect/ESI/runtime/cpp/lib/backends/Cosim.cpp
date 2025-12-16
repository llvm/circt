//===- Cosim.cpp - Connection to ESI simulation via GRPC ------------------===//
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

#include "esi/backends/Cosim.h"
#include "esi/Engines.h"
#include "esi/Services.h"
#include "esi/Utils.h"

#include "cosim.grpc.pb.h"

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>

#include <fstream>
#include <iostream>
#include <set>

using namespace esi;
using namespace esi::cosim;
using namespace esi::services;
using namespace esi::backends::cosim;

using grpc::Channel;
using grpc::ClientContext;
using grpc::ClientReader;
using grpc::ClientReaderWriter;
using grpc::ClientWriter;
using grpc::Status;

static void checkStatus(Status s, const std::string &msg) {
  if (!s.ok())
    throw std::runtime_error(msg + ". Code " + to_string(s.error_code()) +
                             ": " + s.error_message() + " (" +
                             s.error_details() + ")");
}

/// Hack around C++ not having a way to forward declare a nested class.
struct esi::backends::cosim::CosimAccelerator::StubContainer {
  StubContainer(std::unique_ptr<ChannelServer::Stub> stub)
      : stub(std::move(stub)) {}
  std::unique_ptr<ChannelServer::Stub> stub;

  /// Get the type ID for a channel name.
  bool getChannelDesc(const std::string &channelName,
                      esi::cosim::ChannelDesc &desc);
};
using StubContainer = esi::backends::cosim::CosimAccelerator::StubContainer;

/// Parse the connection std::string and instantiate the accelerator. Support
/// the traditional 'host:port' syntax and a path to 'cosim.cfg' which is output
/// by the cosimulation when it starts (which is useful when it chooses its own
/// port).
std::unique_ptr<AcceleratorConnection>
CosimAccelerator::connect(Context &ctxt, std::string connectionString) {
  std::string portStr;
  std::string host = "localhost";

  size_t colon;
  if ((colon = connectionString.find(':')) != std::string::npos) {
    portStr = connectionString.substr(colon + 1);
    host = connectionString.substr(0, colon);
  } else if (connectionString.ends_with("cosim.cfg")) {
    std::ifstream cfg(connectionString);
    std::string line, key, value;

    while (getline(cfg, line))
      if ((colon = line.find(":")) != std::string::npos) {
        key = line.substr(0, colon);
        value = line.substr(colon + 1);
        if (key == "port")
          portStr = value;
        else if (key == "host")
          host = value;
      }

    if (portStr.size() == 0)
      throw std::runtime_error("port line not found in file");
  } else if (connectionString == "env") {
    char *hostEnv = getenv("ESI_COSIM_HOST");
    if (hostEnv)
      host = hostEnv;
    else
      host = "localhost";
    char *portEnv = getenv("ESI_COSIM_PORT");
    if (portEnv)
      portStr = portEnv;
    else
      throw std::runtime_error("ESI_COSIM_PORT environment variable not set");
  } else {
    throw std::runtime_error("Invalid connection std::string '" +
                             connectionString + "'");
  }
  uint16_t port = stoul(portStr);
  auto conn = make_unique<CosimAccelerator>(ctxt, host, port);

  // Using the MMIO manifest method is really only for internal debugging, so it
  // doesn't need to be part of the connection string.
  char *manifestMethod = getenv("ESI_COSIM_MANIFEST_MMIO");
  if (manifestMethod != nullptr)
    conn->setManifestMethod(ManifestMethod::MMIO);

  return conn;
}

/// Construct and connect to a cosim server.
CosimAccelerator::CosimAccelerator(Context &ctxt, std::string hostname,
                                   uint16_t port)
    : AcceleratorConnection(ctxt) {
  // Connect to the simulation.
  auto channel = grpc::CreateChannel(hostname + ":" + std::to_string(port),
                                     grpc::InsecureChannelCredentials());
  rpcClient = new StubContainer(ChannelServer::NewStub(channel));
}
CosimAccelerator::~CosimAccelerator() {
  disconnect();
  if (rpcClient)
    delete rpcClient;
  channels.clear();
}

namespace {
class CosimSysInfo : public SysInfo {
public:
  CosimSysInfo(CosimAccelerator &conn, ChannelServer::Stub *rpcClient)
      : SysInfo(conn), rpcClient(rpcClient) {}

  uint32_t getEsiVersion() const override {
    ::esi::cosim::Manifest response = getManifest();
    return response.esi_version();
  }

  std::vector<uint8_t> getCompressedManifest() const override {
    ::esi::cosim::Manifest response = getManifest();
    std::string compressedManifestStr = response.compressed_manifest();
    return std::vector<uint8_t>(compressedManifestStr.begin(),
                                compressedManifestStr.end());
  }

private:
  ::esi::cosim::Manifest getManifest() const {
    ::esi::cosim::Manifest response;
    // To get around the a race condition where the manifest may not be set yet,
    // loop until it is. TODO: fix this with the DPI API change.
    do {
      ClientContext context;
      VoidMessage arg;
      Status s = rpcClient->GetManifest(&context, arg, &response);
      checkStatus(s, "Failed to get manifest");
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (response.esi_version() < 0);
    return response;
  }

  esi::cosim::ChannelServer::Stub *rpcClient;
};
} // namespace

namespace {
/// Cosim client implementation of a write channel port.
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(AcceleratorConnection &conn,
                        ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                        const Type *type, std::string name)
      : WriteChannelPort(type), conn(conn), rpcClient(rpcClient), desc(desc),
        name(name) {}
  ~WriteCosimChannelPort() = default;

  void connectImpl(const ChannelPort::ConnectOptions &options) override {
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_SERVER)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to server channel");
    assert(desc.name() == name);
  }

protected:
  /// Send a write message to the server.
  void writeImpl(const MessageData &data) override {
    // Add trace logging before sending the message.
    conn.getLogger().trace(
        [this,
         &data](std::string &subsystem, std::string &msg,
                std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_write";
          msg = "Writing message to channel '" + name + "'";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = name;
          (*details)["data_size"] = data.getSize();
          (*details)["message_data"] = data.toHex();
        });

    ClientContext context;
    AddressedMessage msg;
    msg.set_channel_name(name);
    msg.mutable_message()->set_data(data.getBytes(), data.getSize());
    VoidMessage response;
    grpc::Status sendStatus = rpcClient->SendToServer(&context, msg, &response);
    if (!sendStatus.ok())
      throw std::runtime_error("Failed to write to channel '" + name +
                               "': " + std::to_string(sendStatus.error_code()) +
                               " " + sendStatus.error_message() +
                               ". Details: " + sendStatus.error_details());
  }

  bool tryWriteImpl(const MessageData &data) override {
    writeImpl(data);
    return true;
  }

  AcceleratorConnection &conn;
  ChannelServer::Stub *rpcClient;
  /// The channel description as provided by the server.
  ChannelDesc desc;
  /// The name of the channel from the manifest.
  std::string name;
};
} // namespace

namespace {
/// Cosim client implementation of a read channel port. Since gRPC read protocol
/// streams messages back, this implementation is quite complex.
class ReadCosimChannelPort
    : public ReadChannelPort,
      public grpc::ClientReadReactor<esi::cosim::Message> {
public:
  ReadCosimChannelPort(AcceleratorConnection &conn,
                       ChannelServer::Stub *rpcClient, const ChannelDesc &desc,
                       const Type *type, std::string name)
      : ReadChannelPort(type), conn(conn), rpcClient(rpcClient), desc(desc),
        name(name), context(nullptr) {}
  virtual ~ReadCosimChannelPort() { disconnect(); }

  void connectImpl(const ChannelPort::ConnectOptions &options) override {
    // Sanity checking.
    if (desc.dir() != ChannelDesc::Direction::ChannelDesc_Direction_TO_CLIENT)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to client channel");
    assert(desc.name() == name);

    // Initiate a stream of messages from the server.
    if (context)
      return;
    context = new ClientContext();
    rpcClient->async()->ConnectToClientChannel(context, &desc, this);
    StartCall();
    StartRead(&incomingMessage);
  }

  /// Gets called when there's a new message from the server. It'll be stored in
  /// `incomingMessage`.
  void OnReadDone(bool ok) override {
    if (!ok)
      // This happens when we are disconnecting since we are canceling the call.
      return;

    // Read the delivered message and push it onto the queue.
    const std::string &messageString = incomingMessage.data();
    MessageData data(reinterpret_cast<const uint8_t *>(messageString.data()),
                     messageString.size());

    // Add trace logging for the received message.
    conn.getLogger().trace(
        [this,
         &data](std::string &subsystem, std::string &msg,
                std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_read";
          msg = "Received message from channel '" + name + "'";
          details = std::make_unique<std::map<std::string, std::any>>();
          (*details)["channel"] = name;
          (*details)["data_size"] = data.getSize();
          (*details)["message_data"] = data.toHex();
        });

    while (!callback(data))
      // Blocking here could cause deadlocks in specific situations.
      // TODO: Implement a way to handle this better.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Log the message consumption.
    conn.getLogger().trace(
        [this](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_read";
          msg = "Message from channel '" + name + "' consumed";
        });

    // Initiate the next read.
    StartRead(&incomingMessage);
  }

  /// Disconnect this channel from the server.
  void disconnect() override {
    Logger &logger = conn.getLogger();
    logger.debug("cosim_read", "Disconnecting channel " + name);
    if (!context)
      return;
    context->TryCancel();
    // Don't delete the context since gRPC still hold a reference to it.
    // TODO: figure out how to delete it.
    ReadChannelPort::disconnect();
  }

protected:
  AcceleratorConnection &conn;
  ChannelServer::Stub *rpcClient;
  /// The channel description as provided by the server.
  ChannelDesc desc;
  /// The name of the channel from the manifest.
  std::string name;

  ClientContext *context;
  /// Storage location for the incoming message.
  esi::cosim::Message incomingMessage;
};

} // namespace

/// Get the channel description for a channel name. Iterate through the list
/// each time. Since this will only be called a small number of times on a small
/// list, it's not worth doing anything fancy.
bool StubContainer::getChannelDesc(const std::string &channelName,
                                   ChannelDesc &desc) {
  ClientContext context;
  VoidMessage arg;
  ListOfChannels response;
  Status s = stub->ListChannels(&context, arg, &response);
  checkStatus(s, "Failed to list channels");
  for (const auto &channel : response.channels())
    if (channel.name() == channelName) {
      desc = channel;
      return true;
    }
  return false;
}

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(CosimAccelerator &conn, Context &ctxt, const AppIDPath &idPath,
            StubContainer *rpcClient, const HWClientDetails &clients)
      : MMIO(conn, idPath, clients) {
    // We have to locate the channels ourselves since this service might be used
    // to retrieve the manifest.
    ChannelDesc cmdArg, cmdResp;
    if (!rpcClient->getChannelDesc("__cosim_mmio_read_write.arg", cmdArg) ||
        !rpcClient->getChannelDesc("__cosim_mmio_read_write.result", cmdResp))
      throw std::runtime_error("Could not find MMIO channels");

    const esi::Type *i64Type = getType(ctxt, new UIntType(cmdResp.type(), 64));
    const esi::Type *cmdType =
        getType(ctxt, new StructType(cmdArg.type(),
                                     {{"write", new BitsType("i1", 1)},
                                      {"offset", new UIntType("ui32", 32)},
                                      {"data", new BitsType("i64", 64)}}));

    // Get ports, create the function, then connect to it.
    cmdArgPort = std::make_unique<WriteCosimChannelPort>(
        conn, rpcClient->stub.get(), cmdArg, cmdType,
        "__cosim_mmio_read_write.arg");
    cmdRespPort = std::make_unique<ReadCosimChannelPort>(
        conn, rpcClient->stub.get(), cmdResp, i64Type,
        "__cosim_mmio_read_write.result");
    auto *bundleType = new BundleType(
        "cosimMMIO", {{"arg", BundleType::Direction::To, cmdType},
                      {"result", BundleType::Direction::From, i64Type}});
    cmdMMIO.reset(FuncService::Function::get(AppID("__cosim_mmio"), bundleType,
                                             *cmdArgPort, *cmdRespPort));
    cmdMMIO->connect();
  }

#pragma pack(push, 1)
  struct MMIOCmd {
    uint64_t data;
    uint32_t offset;
    bool write;
  };
#pragma pack(pop)

  // Call the read function and wait for a response.
  uint64_t read(uint32_t addr) const override {
    MMIOCmd cmd{.data = 0, .offset = addr, .write = false};
    auto arg = MessageData::from(cmd);
    std::future<MessageData> result = cmdMMIO->call(arg);
    result.wait();
    uint64_t ret = *result.get().as<uint64_t>();
    conn.getLogger().trace(
        [addr, ret](std::string &subsystem, std::string &msg,
                    std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_mmio";
          msg = "MMIO[0x" + toHex(addr) + "] = 0x" + toHex(ret);
        });
    return ret;
  }

  void write(uint32_t addr, uint64_t data) override {
    conn.getLogger().trace(
        [addr,
         data](std::string &subsystem, std::string &msg,
               std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "cosim_mmio";
          msg = "MMIO[0x" + toHex(addr) + "] <- 0x" + toHex(data);
        });
    MMIOCmd cmd{.data = data, .offset = addr, .write = true};
    auto arg = MessageData::from(cmd);
    std::future<MessageData> result = cmdMMIO->call(arg);
    result.wait();
  }

private:
  const esi::Type *getType(Context &ctxt, esi::Type *type) {
    if (auto t = ctxt.getType(type->getID())) {
      delete type;
      return *t;
    }
    ctxt.registerType(type);
    return type;
  }
  std::unique_ptr<WriteCosimChannelPort> cmdArgPort;
  std::unique_ptr<ReadCosimChannelPort> cmdRespPort;
  std::unique_ptr<FuncService::Function> cmdMMIO;
};

#pragma pack(push, 1)
struct HostMemReadReq {
  uint8_t tag;
  uint32_t length;
  uint64_t address;
};

struct HostMemReadResp {
  uint64_t data;
  uint8_t tag;
};

struct HostMemWriteReq {
  uint8_t valid_bytes;
  uint64_t data;
  uint8_t tag;
  uint64_t address;
};

using HostMemWriteResp = uint8_t;
#pragma pack(pop)

class CosimHostMem : public HostMem {
public:
  CosimHostMem(AcceleratorConnection &acc, Context &ctxt,
               StubContainer *rpcClient)
      : HostMem(acc), acc(acc), ctxt(ctxt), rpcClient(rpcClient) {}

  void start() override {
    // We have to locate the channels ourselves since this service might be used
    // to retrieve the manifest.

    if (writeRespPort)
      return;

    // TODO: The types here are WRONG. They need to be wrapped in Channels! Fix
    // this in a subsequent PR.

    // Setup the read side callback.
    ChannelDesc readArg, readResp;
    if (!rpcClient->getChannelDesc("__cosim_hostmem_read_req.data", readArg) ||
        !rpcClient->getChannelDesc("__cosim_hostmem_read_resp.data", readResp))
      throw std::runtime_error("Could not find HostMem read channels");

    const esi::Type *readRespType =
        getType(ctxt, new StructType(readResp.type(),
                                     {{"tag", new UIntType("ui8", 8)},
                                      {"data", new BitsType("i64", 64)}}));
    const esi::Type *readReqType =
        getType(ctxt, new StructType(readArg.type(),
                                     {{"address", new UIntType("ui64", 64)},
                                      {"length", new UIntType("ui32", 32)},
                                      {"tag", new UIntType("ui8", 8)}}));

    // Get ports. Unfortunately, we can't model this as a callback since there
    // will sometimes be multiple responses per request.
    readRespPort = std::make_unique<WriteCosimChannelPort>(
        conn, rpcClient->stub.get(), readResp, readRespType,
        "__cosim_hostmem_read_resp.data");
    readReqPort = std::make_unique<ReadCosimChannelPort>(
        conn, rpcClient->stub.get(), readArg, readReqType,
        "__cosim_hostmem_read_req.data");
    readReqPort->connect(
        [this](const MessageData &req) { return serviceRead(req); });

    // Setup the write side callback.
    ChannelDesc writeArg, writeResp;
    if (!rpcClient->getChannelDesc("__cosim_hostmem_write.arg", writeArg) ||
        !rpcClient->getChannelDesc("__cosim_hostmem_write.result", writeResp))
      throw std::runtime_error("Could not find HostMem write channels");

    const esi::Type *writeRespType =
        getType(ctxt, new UIntType(writeResp.type(), 8));
    const esi::Type *writeReqType =
        getType(ctxt, new StructType(writeArg.type(),
                                     {{"address", new UIntType("ui64", 64)},
                                      {"tag", new UIntType("ui8", 8)},
                                      {"data", new BitsType("i64", 64)}}));

    // Get ports, create the function, then connect to it.
    writeRespPort = std::make_unique<WriteCosimChannelPort>(
        conn, rpcClient->stub.get(), writeResp, writeRespType,
        "__cosim_hostmem_write.result");
    writeReqPort = std::make_unique<ReadCosimChannelPort>(
        conn, rpcClient->stub.get(), writeArg, writeReqType,
        "__cosim_hostmem_write.arg");
    auto *bundleType = new BundleType(
        "cosimHostMem",
        {{"arg", BundleType::Direction::To, writeReqType},
         {"result", BundleType::Direction::From, writeRespType}});
    write.reset(CallService::Callback::get(acc, AppID("__cosim_hostmem_write"),
                                           bundleType, *writeRespPort,
                                           *writeReqPort));
    write->connect([this](const MessageData &req) { return serviceWrite(req); },
                   true);
  }

  // Service the read request as a callback. Simply reads the data from the
  // location specified. TODO: check that the memory has been mapped.
  bool serviceRead(const MessageData &reqBytes) {
    const HostMemReadReq *req = reqBytes.as<HostMemReadReq>();
    acc.getLogger().trace(
        [&](std::string &subsystem, std::string &msg,
            std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "hostmem";
          msg = "Read request: addr=0x" + toHex(req->address) +
                " len=" + std::to_string(req->length) +
                " tag=" + std::to_string(req->tag);
        });
    // Send one response per 8 bytes.
    uint64_t *dataPtr = reinterpret_cast<uint64_t *>(req->address);
    for (uint32_t i = 0, e = (req->length + 7) / 8; i < e; ++i) {
      HostMemReadResp resp{.data = dataPtr[i], .tag = req->tag};
      acc.getLogger().trace(
          [&](std::string &subsystem, std::string &msg,
              std::unique_ptr<std::map<std::string, std::any>> &details) {
            subsystem = "HostMem";
            msg = "Read result: data=0x" + toHex(resp.data) +
                  " tag=" + std::to_string(resp.tag);
          });
      readRespPort->write(MessageData::from(resp));
    }
    return true;
  }

  // Service a write request as a callback. Simply write the data to the
  // location specified. TODO: check that the memory has been mapped.
  MessageData serviceWrite(const MessageData &reqBytes) {
    const HostMemWriteReq *req = reqBytes.as<HostMemWriteReq>();
    acc.getLogger().trace(
        [&](std::string &subsystem, std::string &msg,
            std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "hostmem";
          msg = "Write request: addr=0x" + toHex(req->address) + " data=0x" +
                toHex(req->data) +
                " valid_bytes=" + std::to_string(req->valid_bytes) +
                " tag=" + std::to_string(req->tag);
        });
    uint8_t *dataPtr = reinterpret_cast<uint8_t *>(req->address);
    for (uint8_t i = 0; i < req->valid_bytes; ++i)
      dataPtr[i] = (req->data >> (i * 8)) & 0xFF;
    HostMemWriteResp resp = req->tag;
    return MessageData::from(resp);
  }

  struct CosimHostMemRegion : public HostMemRegion {
    CosimHostMemRegion(std::size_t size) {
      ptr = malloc(size);
      memset(ptr, 0xFF, size);
      this->size = size;
    }
    virtual ~CosimHostMemRegion() { free(ptr); }
    virtual void *getPtr() const override { return ptr; }
    virtual std::size_t getSize() const override { return size; }

  private:
    void *ptr;
    std::size_t size;
  };

  virtual std::unique_ptr<HostMemRegion>
  allocate(std::size_t size, HostMem::Options opts) const override {
    auto ret = std::unique_ptr<HostMemRegion>(new CosimHostMemRegion(size));
    acc.getLogger().debug(
        [&](std::string &subsystem, std::string &msg,
            std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "HostMem";
          msg = "Allocated host memory region at 0x" + toHex(ret->getPtr()) +
                " of size " + std::to_string(size);
        });
    return ret;
  }
  virtual bool mapMemory(void *ptr, std::size_t size,
                         HostMem::Options opts) const override {
    return true;
  }
  virtual void unmapMemory(void *ptr) const override {}

private:
  const esi::Type *getType(Context &ctxt, esi::Type *type) {
    if (auto t = ctxt.getType(type->getID())) {
      delete type;
      return *t;
    }
    ctxt.registerType(type);
    return type;
  }
  AcceleratorConnection &acc;
  Context &ctxt;
  StubContainer *rpcClient;
  std::unique_ptr<WriteCosimChannelPort> readRespPort;
  std::unique_ptr<ReadCosimChannelPort> readReqPort;
  std::unique_ptr<CallService::Callback> read;
  std::unique_ptr<WriteCosimChannelPort> writeRespPort;
  std::unique_ptr<ReadCosimChannelPort> writeReqPort;
  std::unique_ptr<CallService::Callback> write;
};
} // namespace

namespace esi::backends::cosim {
/// Implement the magic cosim channel communication.
class CosimEngine : public Engine {
public:
  CosimEngine(CosimAccelerator &conn, AppIDPath idPath,
              const ServiceImplDetails &details, const HWClientDetails &clients)
      : Engine(conn), conn(conn) {
    // Compute our parents idPath path.
    AppIDPath prefix = std::move(idPath);
    if (prefix.size() > 0)
      prefix.pop_back();

    for (auto client : clients) {
      AppIDPath fullClientPath = prefix + client.relPath;
      std::map<std::string, std::string> channelAssignments;
      for (auto assignment : client.channelAssignments)
        if (assignment.second.type == "cosim")
          channelAssignments[assignment.first] = std::any_cast<std::string>(
              assignment.second.implOptions.at("name"));
      clientChannelAssignments[fullClientPath] = std::move(channelAssignments);
    }
  }

  std::unique_ptr<ChannelPort> createPort(AppIDPath idPath,
                                          const std::string &channelName,
                                          BundleType::Direction dir,
                                          const Type *type) override;

private:
  CosimAccelerator &conn;
  std::map<AppIDPath, std::map<std::string, std::string>>
      clientChannelAssignments;
};
} // namespace esi::backends::cosim

std::unique_ptr<ChannelPort>
CosimEngine::createPort(AppIDPath idPath, const std::string &channelName,
                        BundleType::Direction dir, const Type *type) {

  // Find the client details for the port at 'fullPath'.
  auto f = clientChannelAssignments.find(idPath);
  if (f == clientChannelAssignments.end())
    throw std::runtime_error("Could not find port for '" + idPath.toStr() +
                             "." + channelName + "'");
  const std::map<std::string, std::string> &channelAssignments = f->second;
  auto cosimChannelNameIter = channelAssignments.find(channelName);
  if (cosimChannelNameIter == channelAssignments.end())
    throw std::runtime_error("Could not find channel '" + idPath.toStr() + "." +
                             channelName + "' in cosimulation");

  // Get the endpoint, which may or may not exist. Construct the port.
  // Everything is validated when the client calls 'connect()' on the port.
  ChannelDesc chDesc;
  if (!conn.rpcClient->getChannelDesc(cosimChannelNameIter->second, chDesc))
    throw std::runtime_error("Could not find channel '" + idPath.toStr() + "." +
                             channelName + "' in cosimulation");

  std::unique_ptr<ChannelPort> port;
  std::string cosimChannelName = cosimChannelNameIter->second;
  if (BundlePort::isWrite(dir))
    port = std::make_unique<WriteCosimChannelPort>(
        conn, conn.rpcClient->stub.get(), chDesc, type, cosimChannelName);
  else
    port = std::make_unique<ReadCosimChannelPort>(
        conn, conn.rpcClient->stub.get(), chDesc, type, cosimChannelName);
  return port;
}

void CosimAccelerator::createEngine(const std::string &engineTypeName,
                                    AppIDPath idPath,
                                    const ServiceImplDetails &details,
                                    const HWClientDetails &clients) {

  std::unique_ptr<Engine> engine = nullptr;
  if (engineTypeName == "cosim")
    engine = std::make_unique<CosimEngine>(*this, idPath, details, clients);
  else
    engine = ::esi::registry::createEngine(*this, engineTypeName, idPath,
                                           details, clients);
  registerEngine(idPath, std::move(engine), clients);
}
Service *CosimAccelerator::createService(Service::Type svcType,
                                         AppIDPath idPath, std::string implName,
                                         const ServiceImplDetails &details,
                                         const HWClientDetails &clients) {
  if (svcType == typeid(services::MMIO)) {
    return new CosimMMIO(*this, getCtxt(), idPath, rpcClient, clients);
  } else if (svcType == typeid(services::HostMem)) {
    return new CosimHostMem(*this, getCtxt(), rpcClient);
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(*this, rpcClient->stub.get());
    case ManifestMethod::MMIO:
      return new MMIOSysInfo(getService<services::MMIO>());
    }
  }
  return nullptr;
}

void CosimAccelerator::setManifestMethod(ManifestMethod method) {
  manifestMethod = method;
}

REGISTER_ACCELERATOR("cosim", backends::cosim::CosimAccelerator);
