#include "loopback/LoopbackIP.h"

#include "probe_runner.h"

#include "esi/Accelerator.h"
#include "esi/Manifest.h"
#include "esi/Services.h"
#include "esi/TypedPorts.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace esi;

static int runLoopbackI8(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *inPort = accel->resolvePort(
      {AppID("loopback_inst", 0), AppID("loopback_tohw")}, lastLookup);
  if (!inPort)
    throw std::runtime_error("No loopback_tohw port found");
  BundlePort *outPort = accel->resolvePort(
      {AppID("loopback_inst", 0), AppID("loopback_fromhw")}, lastLookup);
  if (!outPort)
    throw std::runtime_error("No loopback_fromhw port found");

  WriteChannelPort &toHw = inPort->getRawWrite("recv");
  ReadChannelPort &fromHw = outPort->getRawRead("send");
  toHw.connect();
  fromHw.connect();

  uint8_t sendVal = 0x5a;
  toHw.write(MessageData(&sendVal, sizeof(sendVal)));

  MessageData recvMsg;
  fromHw.read(recvMsg);
  if (recvMsg.getSize() != sizeof(uint8_t))
    throw std::runtime_error("Unexpected loopback recv size");
  uint8_t got = *recvMsg.as<uint8_t>();
  if (got != sendVal)
    throw std::runtime_error("Loopback byte mismatch");

  std::cout << "loopback_i8 ok: 0x" << std::hex << (int)got << std::dec << "\n";
  return 0;
}

static int runStructFunc(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("structFunc")}, lastLookup);
  if (!port)
    throw std::runtime_error("No structFunc port found");

  auto *func = port->getAs<services::FuncService::Function>();
  if (!func)
    throw std::runtime_error("structFunc not a FuncService::Function");
  func->connect();

  esi_system::ArgStruct arg{};
  arg.a = 0x1234;
  arg.b = static_cast<int8_t>(-7);

  MessageData resMsg = func->call(MessageData::from(arg)).get();
  const auto *res = resMsg.as<esi_system::ResultStruct>();

  int8_t expectedX = static_cast<int8_t>(arg.b + 1);
  if (res->x != expectedX || res->y != arg.b)
    throw std::runtime_error("Struct func result mismatch");

  std::cout << "struct_func ok: b=" << (int)arg.b << " x=" << (int)res->x
            << " y=" << (int)res->y << "\n";
  return 0;
}

static int runOddStructFunc(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("oddStructFunc")}, lastLookup);
  if (!port)
    throw std::runtime_error("No oddStructFunc port found");

  auto *func = port->getAs<services::FuncService::Function>();
  if (!func)
    throw std::runtime_error("oddStructFunc not a FuncService::Function");
  func->connect();

  esi_system::OddStruct arg{};
  arg.a = 0xabc;
  arg.b = static_cast<int8_t>(-17);
  arg.inner.p = 5;
  arg.inner.q = static_cast<int8_t>(-7);
  arg.inner.r[0] = 3;
  arg.inner.r[1] = 4;

  MessageData resMsg = func->call(MessageData::from(arg)).get();
  const auto *res = resMsg.as<esi_system::OddStruct>();

  uint16_t expectA = static_cast<uint16_t>(arg.a + 1);
  int8_t expectB = static_cast<int8_t>(arg.b - 3);
  uint8_t expectP = static_cast<uint8_t>(arg.inner.p + 5);
  int8_t expectQ = static_cast<int8_t>(arg.inner.q + 2);
  uint8_t expectR0 = static_cast<uint8_t>(arg.inner.r[0] + 1);
  uint8_t expectR1 = static_cast<uint8_t>(arg.inner.r[1] + 2);
  if (res->a != expectA || res->b != expectB || res->inner.p != expectP ||
      res->inner.q != expectQ || res->inner.r[0] != expectR0 ||
      res->inner.r[1] != expectR1)
    throw std::runtime_error("Odd struct func result mismatch");

  std::cout << "odd_struct_func ok: a=" << res->a << " b=" << (int)res->b
            << " p=" << (int)res->inner.p << " q=" << (int)res->inner.q
            << " r0=" << (int)res->inner.r[0] << " r1=" << (int)res->inner.r[1]
            << "\n";
  return 0;
}

static int runArrayFunc(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("arrayFunc")}, lastLookup);
  if (!port)
    throw std::runtime_error("No arrayFunc port found");

  auto *func = port->getAs<services::FuncService::Function>();
  if (!func)
    throw std::runtime_error("arrayFunc not a FuncService::Function");
  func->connect();

  int8_t argArray[1] = {static_cast<int8_t>(-3)};
  MessageData resMsg =
      func->call(MessageData(reinterpret_cast<const uint8_t *>(argArray),
                             sizeof(argArray)))
          .get();

  const auto *res = resMsg.as<esi_system::ResultArray>();
  int8_t a = (*res)[0];
  int8_t b = (*res)[1];
  int8_t expect0 = argArray[0];
  int8_t expect1 = static_cast<int8_t>(argArray[0] + 1);

  bool ok = (a == expect0 && b == expect1) || (a == expect1 && b == expect0);
  if (!ok)
    throw std::runtime_error("Array func result mismatch");

  int8_t low = a;
  int8_t high = b;
  if (low > high) {
    int8_t tmp = low;
    low = high;
    high = tmp;
  }
  std::cout << "array_func ok: " << (int)low << " " << (int)high << "\n";
  return 0;
}

//
// SerialCoordTranslator test
//

struct Coord {
  uint32_t y; // SV ordering: last declared field first in memory
  uint32_t x;
};
#pragma pack(push, 1)
struct SerialCoordHeader {
  uint16_t coordsCount;
  uint32_t yTranslation;
  uint32_t xTranslation;
};
static_assert(sizeof(SerialCoordHeader) == 10, "Size mismatch");
struct SerialCoordData {
  SerialCoordData(uint32_t x, uint32_t y) : _pad_head(0), y(y), x(x) {}
  uint16_t _pad_head;
  uint32_t y;
  uint32_t x;
};
static_assert(sizeof(SerialCoordData) == sizeof(SerialCoordHeader),
              "Size mismatch");
#pragma pack(pop)

// Note: this application is intended to test hardware. As such, we need
// to be able to send batches. So this is not the typical way one would define a
// message struct. It's closer to a streaming style.
struct SerialCoordInput : SegmentedMessageData {
private:
  SerialCoordHeader header;
  std::vector<SerialCoordData> coords;
  SerialCoordHeader footer;

public:
  SerialCoordInput(uint32_t xTrans, uint32_t yTrans,
                   std::vector<SerialCoordData> &coords)
      : coords(coords) {
    header.coordsCount = (uint16_t)coords.size();
    header.xTranslation = xTrans;
    header.yTranslation = yTrans;
    footer.coordsCount = 0;
  }

  size_t numSegments() const override { return 3; }
  Segment segment(size_t idx) const override {
    if (idx == 0)
      return {reinterpret_cast<const uint8_t *>(&header), sizeof(header)};
    else if (idx == 1)
      return {reinterpret_cast<const uint8_t *>(coords.data()),
              coords.size() * sizeof(SerialCoordData)};
    else if (idx == 2)
      return {reinterpret_cast<const uint8_t *>(&footer), sizeof(footer)};
    else
      throw std::out_of_range("SerialCoordInput: invalid segment index");
  }
};

#pragma pack(push, 1)
struct SerialCoordOutputHeader {
  uint8_t _pad[6];
  uint16_t coordsCount;
};
struct SerialCoordOutputData {
  uint32_t y;
  uint32_t x;
};
union SerialCoordOutputFrame {
  SerialCoordOutputHeader header;
  SerialCoordOutputData data;
};
#pragma pack(pop)
static_assert(sizeof(SerialCoordOutputFrame) == 8, "Size mismatch");

static int serialCoordTranslateTest(Accelerator *accel) {
  size_t numCoords = 100;
  uint32_t xTrans = 10, yTrans = 20;

  // Generate random coordinates.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  std::vector<Coord> inputCoords;
  inputCoords.reserve(numCoords);
  for (uint32_t i = 0; i < numCoords; ++i)
    inputCoords.push_back({dist(rng), dist(rng)});

  auto child = accel->getChildren().find(AppID("coord_translator_serial"));
  if (child == accel->getChildren().end())
    throw std::runtime_error("Serial coord translate test: no "
                             "'coord_translator_serial' child found");

  auto &ports = child->second->getPorts();
  auto portIter = ports.find(AppID("translate_coords_serial"));
  if (portIter == ports.end())
    throw std::runtime_error(
        "Serial coord translate test: no 'translate_coords_serial' port found");

  auto *func = portIter->second.getAs<services::FuncService::Function>();
  if (!func)
    throw std::runtime_error(
        "Serial coord translate test: port is not a FuncService::Function");

  // Drive the raw arg port through TypedWritePort so the segmented batch is
  // packed correctly; drain the raw result port directly. We bypass
  // TypedFunction here because `SerialCoordOutputFrame` is a hand-written
  // union (not a real ESI type) and we don't want the typed result decoder
  // to backpressure on the N+2 reply frames.
  TypedWritePort<SerialCoordInput, /*SkipTypeCheck=*/true> argPort(
      func->getRawWrite("arg"));
  ChannelPort::ConnectOptions argOpts(/*bufferSize=*/std::nullopt,
                                      /*translateMessage=*/false);
  argPort.connect(argOpts);
  ReadChannelPort &rawResult = func->getRawRead("result");
  // The reply contains numCoords + 2 raw frames (one header + N data + one
  // footer). Use an unbounded buffer (bufferSize=0) so the polling queue
  // doesn't backpressure the backend mid-stream.
  ChannelPort::ConnectOptions resultOpts(/*bufferSize=*/0,
                                         /*translateMessage=*/false);
  rawResult.connect(resultOpts);

  std::vector<SerialCoordData> coords;
  for (auto &c : inputCoords)
    coords.emplace_back(c.x, c.y);
  SerialCoordInput batch(xTrans, yTrans, coords);
  argPort.write(batch);

  // The bulk-list reply is one header frame + numCoords data frames + one
  // footer frame. TODO: list results aren't decoded here; we just drain the
  // frames so the backend doesn't backpressure / hang.
  MessageData drained;
  for (size_t i = 0; i < numCoords + 2; ++i)
    rawResult.read(drained);
  std::cout << "serial_coord_translate ok\n";
  return 0;
}

static int runDepthConstant(Accelerator *) {
  std::cout << "depth: 0x" << std::hex << esi_system::LoopbackIP::depth
            << std::dec << "\n";
  std::cout << "depth_constant ok\n";
  return 0;
}

ESI_PROBE_REGISTRY("loopback-cpp",
                   "Loopback cosim test using generated ESI headers.",
                   {"depth_constant", &runDepthConstant},
                   {"loopback_i8", &runLoopbackI8},
                   {"struct_func", &runStructFunc},
                   {"odd_struct_func", &runOddStructFunc},
                   {"array_func", &runArrayFunc},
                   {"serial_coord_translate", &serialCoordTranslateTest}, );
