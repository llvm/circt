#include "loopback/LoopbackIP.h"

#include "esi/Accelerator.h"
#include "esi/CLI.h"
#include "esi/Manifest.h"
#include "esi/Services.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

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

  std::cout << "loopback i8 ok: 0x" << std::hex << (int)got << std::dec << "\n";
}

static void runStructFunc(Accelerator *accel) {
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

  std::cout << "struct func ok: b=" << (int)arg.b << " x=" << (int)res->x
            << " y=" << (int)res->y << "\n";
}

static void runOddStructFunc(Accelerator *accel) {
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

  std::cout << "odd struct func ok: a=" << res->a << " b=" << (int)res->b
            << " p=" << (int)res->inner.p << " q=" << (int)res->inner.q
            << " r0=" << (int)res->inner.r[0] << " r1=" << (int)res->inner.r[1]
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

int main(int argc, const char *argv[]) {
  CliParser cli("loopback-cpp");
  cli.description("Loopback cosim test using generated ESI headers.");
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
    runStructFunc(accel);
    runOddStructFunc(accel);
    runArrayFunc(accel);

    conn->disconnect();
  } catch (std::exception &e) {
    ctxt.getLogger().error("loopback-cpp", e.what());
    conn->disconnect();
    return 1;
  }

  return 0;
}
