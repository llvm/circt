#include "loopback/LoopbackIP.h"

#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/Manifest.h"
#include "esi/Services.h"
#include "esi/TypedPorts.h"

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace esi;

static void runLoopbackI8(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *inPort = accel->resolvePort(
      {AppID("loopback_inst", 0), AppID("loopback_tohw")}, lastLookup);
  if (!inPort)
    throw std::runtime_error("No loopback_tohw port found");
  BundlePort *outPort = accel->resolvePort(
      {AppID("loopback_inst", 0), AppID("loopback_fromhw")}, lastLookup);
  if (!outPort)
    throw std::runtime_error("No loopback_fromhw port found");

  // Use TypedWritePort and TypedReadPort instead of raw channels.
  TypedWritePort<uint8_t> toHw(inPort->getRawWrite("recv"));
  TypedReadPort<uint8_t> fromHw(outPort->getRawRead("send"));
  toHw.connect();
  fromHw.connect();

  uint8_t sendVal = 0x5a;
  toHw.write(sendVal);

  uint8_t got = fromHw.read();
  if (got != sendVal)
    throw std::runtime_error("Loopback byte mismatch");

  std::cout << "loopback i8 ok: 0x" << std::hex << (int)got << std::dec << "\n";
}

static void runStructFunc(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("structFunc")}, lastLookup);
  if (!port)
    throw std::runtime_error("No structFunc port found");

  // Use TypedFunction instead of raw FuncService::Function.
  TypedFunction<esi_system::ArgStruct, esi_system::ResultStruct> func =
      port->getAs<services::FuncService::Function>();
  func.connect();

  esi_system::ArgStruct arg{};
  arg.a = 0x1234;
  arg.b = static_cast<int8_t>(-7);

  esi_system::ResultStruct res = func.call(arg).get();

  int8_t expectedX = static_cast<int8_t>(arg.b + 1);
  if (res.x != expectedX || res.y != arg.b)
    throw std::runtime_error("Struct func result mismatch");

  std::cout << "struct func ok: b=" << (int)arg.b << " x=" << (int)res.x
            << " y=" << (int)res.y << "\n";
}

static void runOddStructFunc(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("oddStructFunc")}, lastLookup);
  if (!port)
    throw std::runtime_error("No oddStructFunc port found");

  // Use TypedFunction with OddStruct for both arg and result.
  TypedFunction<esi_system::OddStruct, esi_system::OddStruct> func =
      port->getAs<services::FuncService::Function>();
  func.connect();

  esi_system::OddStruct arg{};
  arg.a = 0xabc;
  arg.b = static_cast<int8_t>(-17);
  arg.inner.p = 5;
  arg.inner.q = static_cast<int8_t>(-7);
  arg.inner.r[0] = 3;
  arg.inner.r[1] = 4;

  esi_system::OddStruct res = func.call(arg).get();

  uint16_t expectA = static_cast<uint16_t>(arg.a + 1);
  int8_t expectB = static_cast<int8_t>(arg.b - 3);
  uint8_t expectP = static_cast<uint8_t>(arg.inner.p + 5);
  int8_t expectQ = static_cast<int8_t>(arg.inner.q + 2);
  uint8_t expectR0 = static_cast<uint8_t>(arg.inner.r[0] + 1);
  uint8_t expectR1 = static_cast<uint8_t>(arg.inner.r[1] + 2);
  if (res.a != expectA || res.b != expectB || res.inner.p != expectP ||
      res.inner.q != expectQ || res.inner.r[0] != expectR0 ||
      res.inner.r[1] != expectR1)
    throw std::runtime_error("Odd struct func result mismatch");

  std::cout << "odd struct func ok: a=" << res.a << " b=" << (int)res.b
            << " p=" << (int)res.inner.p << " q=" << (int)res.inner.q
            << " r0=" << (int)res.inner.r[0] << " r1=" << (int)res.inner.r[1]
            << "\n";
}

static void runArrayFunc(Accelerator *accel) {
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
  std::cout << "array func ok: " << (int)low << " " << (int)high << "\n";
}

static void runSInt4Loopback(Accelerator *accel) {
  AppIDPath lastLookup;
  BundlePort *port = accel->resolvePort({AppID("sint4Func")}, lastLookup);
  if (!port)
    throw std::runtime_error("No sint4Func port found");

  // Use TypedFunction<int8_t, int8_t> for si4 → si4 loopback.
  // si4 fits in int8_t (width 4 <= 8). Tests sign extension of small widths.
  TypedFunction<int8_t, int8_t> func =
      port->getAs<services::FuncService::Function>();
  func.connect();

  // Test positive value.
  int8_t posArg = 5;
  int8_t posResult = func.call(posArg).get();
  if (posResult != posArg)
    throw std::runtime_error("sint4 loopback positive mismatch: got " +
                             std::to_string(posResult));

  // Test negative value (-3, which is 0x0D in si4 wire format).
  int8_t negArg = -3;
  int8_t negResult = func.call(negArg).get();
  if (negResult != negArg)
    throw std::runtime_error("sint4 loopback negative mismatch: got " +
                             std::to_string(negResult));

  std::cout << "sint4 loopback ok: pos=" << (int)posResult
            << " neg=" << (int)negResult << "\n";
}

//
// SerialCoordTranslator test
//

using SerialCoordInput = esi_system::serial_coord_args;
using SerialCoordValue = SerialCoordInput::value_type;

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

static void serialCoordTranslateTest(Accelerator *accel) {
  size_t numCoords = 100;
  uint32_t xTrans = 10, yTrans = 20;

  // Generate random coordinates.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<uint32_t> dist(0, 1000000);
  std::vector<SerialCoordValue> coords;
  coords.reserve(numCoords);
  for (uint32_t i = 0; i < numCoords; ++i)
    coords.emplace_back(dist(rng), dist(rng));

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

  // Keep the raw result channel here: the serial window reply arrives as
  // multiple frames, while FuncService::Function / TypedFunction only waits
  // for a single result message.
  TypedWritePort<SerialCoordInput, /*SkipTypeCheck=*/true> argPort(
      func->getRawWrite("arg"));
  ReadChannelPort &resultPort = func->getRawRead("result");

  argPort.connect(ChannelPort::ConnectOptions(std::nullopt, false));
  resultPort.connect(ChannelPort::ConnectOptions(std::nullopt, false));

  auto batch = std::make_unique<SerialCoordInput>(xTrans, yTrans, coords);
  argPort.write(batch);

  std::vector<SerialCoordValue> results;
  while (true) {
    MessageData msg;
    resultPort.read(msg);
    if (msg.getSize() != sizeof(SerialCoordOutputFrame))
      throw std::runtime_error("Unexpected result message size");

    const auto *frame =
        reinterpret_cast<const SerialCoordOutputFrame *>(msg.getBytes());
    uint16_t batchCount = frame->header.coordsCount;
    if (batchCount == 0)
      break;

    for (uint16_t i = 0; i < batchCount; ++i) {
      resultPort.read(msg);
      if (msg.getSize() != sizeof(SerialCoordOutputFrame))
        throw std::runtime_error("Unexpected result message size");
      const auto *dFrame =
          reinterpret_cast<const SerialCoordOutputFrame *>(msg.getBytes());
      results.push_back({dFrame->data.y, dFrame->data.x});
    }
  }

  if (results.size() != coords.size())
    throw std::runtime_error("Serial coord translate result size mismatch");
  for (size_t i = 0; i < coords.size(); ++i) {
    uint32_t expX = coords[i].x + xTrans;
    uint32_t expY = coords[i].y + yTrans;
    if (results[i].x != expX || results[i].y != expY)
      throw std::runtime_error("Serial coord translate result mismatch");
  }

  argPort.disconnect();
  resultPort.disconnect();
}

int main(int argc, const char *argv[]) {
  CliParser cli("loopback-typed-cpp");
  cli.description(
      "Loopback cosim test using generated ESI headers and typed ports.");
  if (int rc = cli.esiParse(argc, argv))
    return rc;
  if (!cli.get_help_ptr()->empty())
    return 0;

  Context &ctxt = cli.getContext();
  AcceleratorConnection *conn = cli.connect();
  try {
    const auto &info = *conn->getService<services::SysInfo>();
    Manifest manifest(ctxt, info.getJsonManifest());
    Accelerator *accel = manifest.buildAccelerator(*conn);
    conn->getServiceThread()->addPoll(*accel);

    std::cout << "depth: 0x" << std::hex << esi_system::LoopbackIP::depth
              << std::dec << "\n";

    runLoopbackI8(accel);
    runSInt4Loopback(accel);
    runStructFunc(accel);
    runOddStructFunc(accel);
    runArrayFunc(accel);
    serialCoordTranslateTest(accel);

    conn->disconnect();
  } catch (std::exception &e) {
    ctxt.getLogger().error("loopback-typed-cpp", e.what());
    conn->disconnect();
    return 1;
  }

  return 0;
}
