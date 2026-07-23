//===- Cosim.cpp - Connection to ESI simulation ---------------------------===//
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
#include "esi/Ports.h"
#include "esi/Services.h"
#include "esi/Utils.h"
#include "esi/backends/RpcClient.h"

#include <array>
#include <cstring>
#include <format>
#include <fstream>
#include <iostream>
#include <set>

using namespace esi;
using namespace esi::services;
using namespace esi::backends::cosim;

namespace {

//===----------------------------------------------------------------------===//
// WriteCosimChannelPort
//===----------------------------------------------------------------------===//

/// Cosim client implementation of a write channel port.
class WriteCosimChannelPort : public WriteChannelPort {
public:
  WriteCosimChannelPort(AcceleratorConnection &conn, RpcClient &client,
                        const RpcClient::ChannelDesc &desc, const Type *type,
                        std::string name)
      : WriteChannelPort(type), conn(conn), client(client), desc(desc),
        name(std::move(name)) {}
  ~WriteCosimChannelPort() = default;

  void connectImpl(const ChannelPort::ConnectOptions &options) override {
    if (desc.dir != RpcClient::ChannelDirection::ToServer)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to server channel");
  }

protected:
  void writeImpl(const MessageData &data) override {
    auto frames = getMessageFrames(data);
    for (const auto &frame : frames) {
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

      client.writeToServer(name, frame);
    }
  }
  bool tryWriteImpl(const MessageData &data) override {
    // For simplicity, this implementation does not support backpressure and
    // always returns true. A more complex implementation could track pending
    // messages and return false if there are too many.
    writeImpl(data);
    return true;
  }

private:
  AcceleratorConnection &conn;
  RpcClient &client;
  RpcClient::ChannelDesc desc;
  std::string name;
};

//===----------------------------------------------------------------------===//
// ReadCosimChannelPort
//===----------------------------------------------------------------------===//

/// Cosim client implementation of a read channel port. The wire transport
/// (see `CosimRpc`) delivers messages via callback, so this class just
/// forwards them to the registered `ReadChannelPort` consumer.
class ReadCosimChannelPort : public ReadChannelPort {
public:
  ReadCosimChannelPort(AcceleratorConnection &conn, RpcClient &client,
                       const RpcClient::ChannelDesc &desc, const Type *type,
                       std::string name)
      : ReadChannelPort(type), conn(conn), client(client), desc(desc),
        name(std::move(name)) {}

  ~ReadCosimChannelPort() = default;

  void connectImpl(const ChannelPort::ConnectOptions &options) override {
    if (desc.dir != RpcClient::ChannelDirection::ToClient)
      throw std::runtime_error("Channel '" + name +
                               "' is not a to client channel");

    // Connect to the channel and set up callback.
    connection = client.connectClientReceiver(
        name, [this](std::unique_ptr<SegmentedMessageData> &data) {
          // Add trace logging for the received message.
          conn.getLogger().trace(
              [this, &data](
                  std::string &subsystem, std::string &msg,
                  std::unique_ptr<std::map<std::string, std::any>> &details) {
                subsystem = "cosim_read";
                msg = "Received message from channel '" + name + "'";
                details = std::make_unique<std::map<std::string, std::any>>();
                MessageData flat = data->toMessageData();
                (*details)["channel"] = name;
                (*details)["data_size"] = flat.getSize();
                (*details)["message_data"] = flat.toHex();
              });

          bool consumed = invokeCallback(data);

          if (consumed) {
            // Log the message consumption.
            conn.getLogger().trace(
                [this](
                    std::string &subsystem, std::string &msg,
                    std::unique_ptr<std::map<std::string, std::any>> &details) {
                  subsystem = "cosim_read";
                  msg = "Message from channel '" + name + "' consumed";
                });
          }

          return consumed;
        });
  }

  void disconnect() override {
    conn.getLogger().debug("cosim_read", "Disconnecting channel " + name);
    if (connection) {
      connection->disconnect();
      connection.reset();
    }
    ReadChannelPort::disconnect();
  }

private:
  AcceleratorConnection &conn;
  RpcClient &client;
  RpcClient::ChannelDesc desc;
  std::string name;
  std::unique_ptr<RpcClient::ReadChannelConnection> connection;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// CosimAccelerator
//===----------------------------------------------------------------------===//

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
  rpcClient = std::make_unique<RpcClient>(getLogger(), hostname, port);
}
CosimAccelerator::~CosimAccelerator() {
  disconnect();
  clearOwnedObjects();
  channels.clear();
}

namespace {
class CosimSysInfo : public SysInfo {
#pragma pack(push, 1)
  struct CycleInfo {
    uint64_t freq;
    uint64_t cycle;
  };
#pragma pack(pop)

public:
  CosimSysInfo(CosimAccelerator &conn, RpcClient *rpcClient)
      : SysInfo(conn), rpcClient(rpcClient) {
    // This is an optional interface; if the channels aren't present, we simply
    // report no cycle/frequency information.
    RpcClient::ChannelDesc argDesc, resultDesc;
    if (!rpcClient->getChannelDesc("__cosim_cycle_count.arg", argDesc) ||
        !rpcClient->getChannelDesc("__cosim_cycle_count.result", resultDesc))
      return;

    Context &ctxt = conn.getCtxt();
    const esi::Type *i1Type = getType(ctxt, new BitsType("i1", 1));
    const esi::Type *i64Type = getType(ctxt, new BitsType("i64", 64));
    const esi::Type *resultType =
        getType(ctxt, new StructType(resultDesc.type,
                                     {{"cycle", i64Type}, {"freq", i64Type}}));

    reqPort = std::make_unique<WriteCosimChannelPort>(
        conn, *rpcClient, argDesc, i1Type, "__cosim_cycle_count.arg");
    respPort = std::make_unique<ReadCosimChannelPort>(
        conn, *rpcClient, resultDesc, resultType, "__cosim_cycle_count.result");
    auto *bundleType =
        new BundleType("cosimCycleCount",
                       {{"arg", BundleType::Direction::To, i1Type},
                        {"result", BundleType::Direction::From, resultType}});
    func.reset(FuncService::Function::get(AppID("__cosim_cycle_count"),
                                          bundleType, *reqPort, *respPort));
    func->connect();
  }

  uint32_t getEsiVersion() const override { return rpcClient->getEsiVersion(); }
  std::optional<uint64_t> getCycleCount() const override {
    if (!func)
      return std::nullopt;
    return getCycleInfo().cycle;
  }
  std::optional<uint64_t> getCoreClockFrequency() const override {
    if (!func)
      return std::nullopt;
    return getCycleInfo().freq;
  }

  std::vector<uint8_t> getCompressedManifest() const override {
    return rpcClient->getCompressedManifest();
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

  RpcClient *rpcClient;
  std::unique_ptr<WriteCosimChannelPort> reqPort;
  std::unique_ptr<ReadCosimChannelPort> respPort;
  std::unique_ptr<FuncService::Function> func;

  CycleInfo getCycleInfo() const {
    MessageData arg({1}); // 1-bit trigger message
    std::future<MessageData> result = func->call(arg);
    result.wait();
    MessageData respMsg = result.get();
    return *respMsg.as<CycleInfo>();
  }
};
} // namespace

namespace {
class CosimMMIO : public MMIO {
public:
  CosimMMIO(CosimAccelerator &conn, Context &ctxt, const AppIDPath &idPath,
            RpcClient *rpcClient, const HWClientDetails &clients)
      : MMIO(conn, idPath, clients) {
    // We have to locate the channels ourselves since this service might be used
    // to retrieve the manifest.
    RpcClient::ChannelDesc cmdArg, cmdResp;
    if (!rpcClient->getChannelDesc("__cosim_mmio_read_write.arg", cmdArg) ||
        !rpcClient->getChannelDesc("__cosim_mmio_read_write.result", cmdResp))
      throw std::runtime_error("Could not find MMIO channels");

    const esi::Type *i64Type = getType(ctxt, new UIntType(cmdResp.type, 64));
    const esi::Type *cmdType = getType(
        ctxt, new StructType(cmdArg.type, {{"write", new BitsType("i1", 1)},
                                           {"offset", new UIntType("ui32", 32)},
                                           {"data", new BitsType("i64", 64)}}));

    // Get ports, create the function, then connect to it.
    cmdArgPort = std::make_unique<WriteCosimChannelPort>(
        conn, *rpcClient, cmdArg, cmdType, "__cosim_mmio_read_write.arg");
    cmdRespPort = std::make_unique<ReadCosimChannelPort>(
        conn, *rpcClient, cmdResp, i64Type, "__cosim_mmio_read_write.result");
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

using HostMemWriteResp = uint8_t;
#pragma pack(pop)

// ESI lowers a parallel-window frame MSB-first (the first-declared field
// occupies the highest bits) into a little-endian byte buffer; array elements
// are packed least-index-first (element[0] in the low bits). The sub-byte
// `last` / `data_size` fields make these frames bit-packed and not
// byte-aligned, so -- to stay ABI-portable across toolchains (bit-field and
// sub-byte `#pragma pack` layout is not guaranteed under MSVC) -- each frame is
// a plain byte array whose bits are assembled/extracted explicitly with these
// helpers. `bitOff` counts from the LSB (bit 0 of byte 0).
void putBits(uint8_t *buf, size_t bitOff, size_t width, uint64_t val) {
  for (size_t i = 0; i < width; ++i)
    if ((val >> i) & 1ULL)
      buf[(bitOff + i) >> 3] |= static_cast<uint8_t>(1u << ((bitOff + i) & 7));
}
uint64_t getBits(const uint8_t *buf, size_t bitOff, size_t width) {
  uint64_t val = 0;
  for (size_t i = 0; i < width; ++i)
    if (buf[(bitOff + i) >> 3] & (1u << ((bitOff + i) & 7)))
      val |= (1ULL << i);
  return val;
}

// Read-response frame: one bus beat of a burst read. The client-facing read
// response is a parallel window over struct{tag, data: list<i<HostMemWidth>>};
// with one word per beat (num_items=1) the lowered frame is
//   struct{tag: ui8, data: i64, last: i1}    (73 bits)
// packed MSB-first: tag | data | last. `last` marks the final beat of a burst.
class HostMemReadRespFrame {
public:
  static constexpr size_t kMessageBits = 8 + 64 + 1;              // 73
  static constexpr size_t kMessageBytes = (kMessageBits + 7) / 8; // 10

  HostMemReadRespFrame(uint8_t tag, uint64_t data, bool last) {
    putBits(bytes.data(), kLastOff, kLastW, last ? 1 : 0);
    putBits(bytes.data(), kDataOff, kDataW, data);
    putBits(bytes.data(), kTagOff, kTagW, tag);
  }

  uint8_t tag() const { return getBits(bytes.data(), kTagOff, kTagW); }
  uint64_t data() const { return getBits(bytes.data(), kDataOff, kDataW); }
  bool last() const { return getBits(bytes.data(), kLastOff, kLastW) != 0; }

  MessageData toMessage() const {
    return MessageData(bytes.data(), bytes.size());
  }

private:
  // MSB-first field order => reverse (LSB-most) bit offsets.
  static constexpr size_t kLastW = 1, kLastOff = 0;                  // bit 0
  static constexpr size_t kDataW = 64, kDataOff = kLastOff + kLastW; // bit 1
  static constexpr size_t kTagW = 8, kTagOff = kDataOff + kDataW;    // bit 65
  std::array<uint8_t, kMessageBytes> bytes{};
};

// Write-request frame: one bus beat of a burst write. The upstream write
// request is a parallel window over struct{address, tag, data: list<i8>} with
// num_items = the host-memory bus width in bytes (cosim HostMemWidth=64 => 8).
// The lowered frame is
//   struct{address: ui64, tag: ui8, data: i8[8], data_size: ui3, last: i1}
// (140 bits) packed MSB-first: address | tag | data[8] | data_size | last, with
// data element[i] in ascending bits. `data_size` holds (valid_bytes - 1);
// `last` marks the final beat. Received from the device, so construct from raw
// bytes plus read accessors.
class HostMemWriteReqFrame {
public:
  static constexpr size_t kNumItems = 8; // bus width in bytes (cosim: 64 / 8)
  static constexpr size_t kMessageBits = 64 + 8 + kNumItems * 8 + 3 + 1; // 140
  static constexpr size_t kMessageBytes = (kMessageBits + 7) / 8;        // 18

  explicit HostMemWriteReqFrame(const uint8_t *raw) {
    std::memcpy(bytes.data(), raw, kMessageBytes);
  }

  uint64_t address() const { return getBits(bytes.data(), kAddrOff, kAddrW); }
  uint8_t tag() const { return getBits(bytes.data(), kTagOff, kTagW); }
  uint8_t dataByte(size_t i) const {
    return getBits(bytes.data(), kDataOff + 8 * i, 8);
  }
  // Number of valid data bytes in this beat (data_size holds valid_bytes - 1).
  unsigned validBytes() const {
    return static_cast<unsigned>(getBits(bytes.data(), kSizeOff, kSizeW)) + 1;
  }
  bool last() const { return getBits(bytes.data(), kLastOff, kLastW) != 0; }

private:
  static constexpr size_t kLastW = 1, kLastOff = 0;
  static constexpr size_t kSizeW = 3, kSizeOff = kLastOff + kLastW;
  static constexpr size_t kDataW = kNumItems * 8, kDataOff = kSizeOff + kSizeW;
  static constexpr size_t kTagW = 8, kTagOff = kDataOff + kDataW;
  static constexpr size_t kAddrW = 64, kAddrOff = kTagOff + kTagW;
  std::array<uint8_t, kMessageBytes> bytes{};
};

// PCIe caps a single memory read request at the Max_Read_Request_Size, whose
// largest encoding (PCIe Gen 4 and earlier) is 4096 bytes, but root ports often
// negotiate a smaller limit. Model a conservative 64-double-word (256-byte) cap
// here: a read request larger than this is a protocol violation.
static constexpr uint32_t kPcieMaxReadRequestBytes = 64 * 4;

class CosimHostMem : public HostMem {
public:
  CosimHostMem(AcceleratorConnection &acc, Context &ctxt, RpcClient *rpcClient)
      : HostMem(acc), acc(acc), ctxt(ctxt), rpcClient(rpcClient) {}

  void start() override {
    // We have to locate the channels ourselves since this service might be used
    // to retrieve the manifest.

    if (writeRespPort)
      return;

    // TODO: The types here are WRONG. They need to be wrapped in Channels! Fix
    // this in a subsequent PR.

    // Setup the read side callback.
    RpcClient::ChannelDesc readArg, readResp;
    if (!rpcClient->getChannelDesc("__cosim_hostmem_read_req.data", readArg) ||
        !rpcClient->getChannelDesc("__cosim_hostmem_read_resp.data", readResp))
      throw std::runtime_error("Could not find HostMem read channels");

    const esi::Type *readRespType = getType(
        ctxt, new StructType(readResp.type, {{"tag", new UIntType("ui8", 8)},
                                             {"data", new BitsType("i64", 64)},
                                             {"last", new BitsType("i1", 1)}}));
    const esi::Type *readReqType =
        getType(ctxt, new StructType(readArg.type,
                                     {{"address", new UIntType("ui64", 64)},
                                      {"length", new UIntType("ui32", 32)},
                                      {"tag", new UIntType("ui8", 8)}}));

    // Get ports. Unfortunately, we can't model this as a callback since there
    // will sometimes be multiple responses per request.
    readRespPort = std::make_unique<WriteCosimChannelPort>(
        conn, *rpcClient, readResp, readRespType,
        "__cosim_hostmem_read_resp.data");
    readReqPort = std::make_unique<ReadCosimChannelPort>(
        conn, *rpcClient, readArg, readReqType,
        "__cosim_hostmem_read_req.data");
    readReqPort->connect(
        [this](const MessageData &req) { return serviceRead(req); });

    // Setup the write side callback.
    RpcClient::ChannelDesc writeArg, writeResp;
    if (!rpcClient->getChannelDesc("__cosim_hostmem_write.arg", writeArg) ||
        !rpcClient->getChannelDesc("__cosim_hostmem_write.result", writeResp))
      throw std::runtime_error("Could not find HostMem write channels");

    const esi::Type *writeRespType =
        getType(ctxt, new UIntType(writeResp.type, 8));
    const esi::Type *writeReqType =
        getType(ctxt, new StructType(writeArg.type,
                                     {{"address", new UIntType("ui64", 64)},
                                      {"tag", new UIntType("ui8", 8)},
                                      {"data", new BitsType("i64", 64)},
                                      {"data_size", new UIntType("ui3", 3)},
                                      {"last", new BitsType("i1", 1)}}));

    // Get ports, create the function, then connect to it.
    writeRespPort = std::make_unique<WriteCosimChannelPort>(
        conn, *rpcClient, writeResp, writeRespType,
        "__cosim_hostmem_write.result");
    writeReqPort = std::make_unique<ReadCosimChannelPort>(
        conn, *rpcClient, writeArg, writeReqType, "__cosim_hostmem_write.arg");
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
    // Send one response per 8 bytes. Zero-length reads (e.g. void / zero-width
    // types) indicates a bug in the hardware and we log an error, but we still
    // send a single response.
    uint64_t *dataPtr = reinterpret_cast<uint64_t *>(req->address);
    uint32_t numDataResps = (req->length + 7) / 8;
    if (numDataResps == 0)
      acc.getLogger().error(
          "hostmem",
          std::format("Read request with length=0 from addr=0x{} tag={}. "
                      "Reads of length 0 are not valid and indicate a bug "
                      "in the requester.",
                      toHex(req->address), req->tag));
    if (req->length > kPcieMaxReadRequestBytes)
      acc.getLogger().error(
          "hostmem",
          std::format("Read request length={} from addr=0x{} tag={} exceeds "
                      "the PCIe maximum read request size ({} bytes). The "
                      "requester must split reads larger than this into "
                      "multiple requests.",
                      req->length, toHex(req->address), req->tag,
                      kPcieMaxReadRequestBytes));
    uint32_t numResps = std::max(numDataResps, 1u);
    for (uint32_t i = 0; i < numResps; ++i) {
      uint64_t word = i < numDataResps ? dataPtr[i] : 0;
      bool last = i + 1 == numResps;
      HostMemReadRespFrame frame(req->tag, word, last);
      acc.getLogger().trace(
          [&](std::string &subsystem, std::string &msg,
              std::unique_ptr<std::map<std::string, std::any>> &details) {
            subsystem = "HostMem";
            msg = "Read result: data=0x" + toHex(word) +
                  " tag=" + std::to_string(req->tag) +
                  " last=" + std::to_string(last);
          });
      readRespPort->write(frame.toMessage());
    }
    return true;
  }

  // Service a write request as a callback. Simply write the data to the
  // location specified. TODO: check that the memory has been mapped.
  MessageData serviceWrite(const MessageData &reqBytes) {
    if (reqBytes.getSize() != HostMemWriteReqFrame::kMessageBytes)
      throw std::runtime_error(
          "HostMem write frame size mismatch. Size is " +
          std::to_string(reqBytes.getSize()) + ", expected " +
          std::to_string(HostMemWriteReqFrame::kMessageBytes) + ".");
    HostMemWriteReqFrame req(reqBytes.getBytes());
    acc.getLogger().trace(
        [&](std::string &subsystem, std::string &msg,
            std::unique_ptr<std::map<std::string, std::any>> &details) {
          subsystem = "hostmem";
          msg = "Write request: addr=0x" + toHex(req.address()) +
                " valid_bytes=" + std::to_string(req.validBytes()) +
                " tag=" + std::to_string(req.tag()) +
                " last=" + std::to_string(req.last());
        });
    uint8_t *dataPtr = reinterpret_cast<uint8_t *>(req.address());
    unsigned validBytes = req.validBytes();
    for (unsigned i = 0; i < validBytes; ++i)
      dataPtr[i] = req.dataByte(i);
    HostMemWriteResp resp = req.tag();
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
  RpcClient *rpcClient;
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
  RpcClient::ChannelDesc chDesc;
  if (!conn.rpcClient->getChannelDesc(cosimChannelNameIter->second, chDesc))
    throw std::runtime_error("Could not find channel '" + idPath.toStr() + "." +
                             channelName + "' in cosimulation");

  std::unique_ptr<ChannelPort> port;
  std::string fullChannelName = idPath.toStr() + "." + channelName;
  if (BundlePort::isWrite(dir))
    port = std::make_unique<WriteCosimChannelPort>(
        conn, *conn.rpcClient, chDesc, type, fullChannelName);
  else
    port = std::make_unique<ReadCosimChannelPort>(conn, *conn.rpcClient, chDesc,
                                                  type, fullChannelName);
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
    return new CosimMMIO(*this, getCtxt(), idPath, rpcClient.get(), clients);
  } else if (svcType == typeid(services::HostMem)) {
    return new CosimHostMem(*this, getCtxt(), rpcClient.get());
  } else if (svcType == typeid(SysInfo)) {
    switch (manifestMethod) {
    case ManifestMethod::Cosim:
      return new CosimSysInfo(*this, rpcClient.get());
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
