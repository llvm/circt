// REQUIRES: esi-cosim

// clang-format off

// Create ESI system
// RUN: rm -rf %t
// RUN: %PYTHON% %S/../esi_ram.py %t 2>&1

// Create ESI CPP API
// ...
// RUN: cp %S/CMakeLists.txt %t
// RUN: cp %s %t
// RUN: cmake -S %t -B %T/build -DCIRCT_DIR=%CIRCT_SOURCE%

// Run test
// RN: esi-cosim-runner.py --tmpdir %t --schema %t/hw/schema.capnp %s %t/hw/*.sv
// RN: ./%T/build/esi_ram_test %t %t/hw/schema.capnp

// clang-format on
#include <any>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <capnp/ez-rpc.h>

#include "hw/schema.capnp.h"

namespace ESICPP {

// Custom type to hold the interface descriptions because i can't for the life
// of me figure out how to cleanly keep capnproto messages around...
struct EsiDpiInterfaceDesc {
  std::string endpointID;
  uint64_t sendTypeID;
  uint64_t recvTypeID;
};

// ESI CPP API goes here. This is general for all backends.

// Bas class for all ports. This will contact the backend to get a port
// handle.
template <typename TBackend>
class Port {
public:
  // std::in_place_type allows for type deduction of the template arguments
  // (template constructor must be able to deduce its template parameters from
  // its arguments).
  template <typename TRead, typename TWrite>
  Port(std::in_place_type_t<TRead>, std::in_place_type_t<TWrite>,
       const std::vector<std::string> &clientPath, TBackend &backend,
       const std::string &implType)
      : backend(&backend) {}

private:
  TBackend *backend = nullptr;
};

template <typename ReadType, typename WriteType, typename TBackend>
class ReadWritePort : public Port<TBackend> {
public:
  using TPort = Port<TBackend>;
  ReadWritePort(const std::vector<std::string> &clientPath, TBackend &backend,
                const std::string &implType)
      : Port<TBackend>(std::in_place_type<ReadType>,
                       std::in_place_type<WriteType>, clientPath, backend,
                       implType) {
    // If a backend doesn't support a particular implementation type, just skip
    // it. We don't want to error out on services which aren't being used.
    if (backend.supportsImpl(implType)) {
      // Create the port
      port =
          backend.template getPort<typename ReadType::BackendType,
                                   typename WriteType::BackendType>(clientPath);
    }
  }

  ReadType operator()(WriteType arg) {
    // something like:
    auto req = port->sendRequest();
    req.setMsg(arg.toCapnp().asReader());
    return ReadType();
  }

  std::optional<typename ::EsiDpiEndpoint<
      typename ReadType::BackendType, typename WriteType::BackendType>::Client>
      port;
};

template <typename WriteType, typename TBackend>
class WritePort : public Port<TBackend> {
public:
  WritePort(const std::vector<std::string> &clientPath, TBackend &backend,
            const std::string &implType)
      : Port<TBackend>(std::in_place_type<void>, std::in_place_type<WriteType>,
                       clientPath, backend, implType) {

    auto ep = backend.template getPort<void, WriteType>(clientPath);
  }

  void operator()(WriteType arg) {
    // something like:
    // return backend(arg.toBackendType());
  }
};

template <typename TBackend>
class Module {
public:
  Module(const std::vector<std::shared_ptr<Port<TBackend>>> &ports)
      : ports(ports) {}

protected:
  std::vector<std::shared_ptr<Port<TBackend>>> ports;
};

} // namespace ESICPP

namespace esi_cosim {
// ESI cosim backend goes here.

template <typename TClient, typename T>
class CosimPort {
public:
  CosimPort(TClient &client, const std::string &epName) {}

  template <typename TMessage>
  bool write(TMessage &msg) {

    // ep.send(msg).wait();
    return true;
  }

  template <typename TMessage>
  TMessage read(TMessage &msg) {
    // ep.recv(msg).wait();
    return true;
  }

private:
};

class Cosim {
public:
  Cosim(const std::string &host, uint64_t hostPort) {
    ezClient = std::make_unique<capnp::EzRpcClient>(host, hostPort);
    dpiClient = std::make_unique<CosimDpiServer::Client>(
        ezClient->getMain<CosimDpiServer>());

    list();
  }

  // Returns a list of all available endpoints.
  const std::vector<ESICPP::EsiDpiInterfaceDesc> &list() {
    if (endpoints.has_value())
      return *endpoints;

    // Query the DPI server for a list of available endpoints.
    auto listReq = dpiClient->listRequest();
    auto ifaces = listReq.send().wait(ezClient->getWaitScope()).getIfaces();
    endpoints = std::vector<ESICPP::EsiDpiInterfaceDesc>();
    for (auto iface : ifaces) {
      ESICPP::EsiDpiInterfaceDesc desc;
      desc.endpointID = iface.getEndpointID().cStr();
      desc.sendTypeID = iface.getSendTypeID();
      desc.recvTypeID = iface.getRecvTypeID();
      endpoints->push_back(desc);
    }

    // print out the endpoints
    for (auto ep : *endpoints) {
      std::cout << "Endpoint: " << ep.endpointID << std::endl;
      std::cout << "  Send Type: " << ep.sendTypeID << std::endl;
      std::cout << "  Recv Type: " << ep.recvTypeID << std::endl;
    }

    return *endpoints;
  }

  template <typename TRead, typename TWrite>
  auto getPort(const std::vector<std::string> &clientPath) {
    // Join client path into a single string with '.' as a separator.
    std::string clientPathStr;
    for (auto &path : clientPath) {
      if (!clientPathStr.empty())
        clientPathStr += '.';
      clientPathStr += path;
    }

    // Everything is nested under "TOP.top"
    clientPathStr = "TOP.top." + clientPathStr;

    auto openReq = dpiClient->openRequest<TRead, TWrite>();

    // Scan through the available endpoints to find the requested one.
    bool found = false;
    for (auto &ep : list()) {
      auto epid = ep.endpointID;
      if (epid == clientPathStr) {
        auto iface = openReq.getIface();
        iface.setEndpointID(epid);
        iface.setSendTypeID(ep.sendTypeID);
        iface.setRecvTypeID(ep.recvTypeID);
        found = true;
        break;
      }
    }

    if (!found) {
      throw std::runtime_error("Could not find endpoint: " + clientPathStr);
    }

    // Open the endpoint.
    auto openResp = openReq.send().wait(ezClient->getWaitScope());
    return openResp.getIface();
  }

  bool supportsImpl(const std::string &implType) {
    // The cosim backend only supports cosim connectivity implementations
    return implType == "cosim";
  }

protected:
  std::unique_ptr<capnp::EzRpcClient> ezClient;
  std::unique_ptr<CosimDpiServer::Client> dpiClient;
  std::optional<std::vector<ESICPP::EsiDpiInterfaceDesc>> endpoints;
};

} // namespace esi_cosim

namespace ESIMem {

// Generated things for the current ESI system.

// "pretty" types to wrap Cap'n Proto madness types. Can easily be autogenerated
// based on the input schema.
struct I1 {
  // Data members.
  bool i;

  I1(bool i) : i(i) {}
  I1() = default;

  // Unary types have convenience conversion operators.
  operator bool() const { return i; }

  // Spaceship operator for comparison convenience.
  auto operator<=>(const I1 &) const = default;

  // Generated sibling type.
  using BackendType = ::I1;
  BackendType::Builder toCapnp() {
    auto cp = BackendType::Builder(capnp::_::StructBuilder());
    cp.setI(i);
    return cp;
  }
};

struct I3 {
  using BackendType = ::I3;
  uint8_t i;

  // Convenience constructor due to unary type (allows implicit conversion from
  // literals, makes the API a bit less verbose).
  I3(uint8_t i) : i(i) {}
  I3() = default;

  operator uint8_t() const { return i; }
  auto operator<=>(const I3 &) const = default;

  BackendType::Builder toCapnp() {
    auto cp = BackendType::Builder(capnp::_::StructBuilder());
    cp.setI(i);
    return cp;
  }
};

struct I64 {
  using BackendType = ::I64;
  uint64_t i;
  // use default constructors for all types.
  I64(uint64_t i) : i(i) {}
  I64(int i) : i(i) {}
  I64() = default;
  auto operator<=>(const I64 &) const = default;

  operator uint64_t() const { return i; }
  BackendType::Builder toCapnp() {
    auto cp = BackendType::Builder(capnp::_::StructBuilder());
    cp.setI(i);
    return cp;
  }
};

struct Struct16871797234873963366 {
  using BackendType = ::Struct16871797234873963366;
  uint8_t address;
  uint64_t data;

  auto operator<=>(const Struct16871797234873963366 &) const = default;

  BackendType::Builder toCapnp() {
    auto cp = BackendType::Builder(capnp::_::StructBuilder());
    cp.setAddress(address);
    cp.setData(data);
    return cp;
  }
};

template <typename TBackend>
class MemComms : public ESICPP::Module<TBackend> {
  using Port = ESICPP::Port<TBackend>;

public:
  // Port type declarations
  using Tread0 = ESICPP::ReadWritePort</*readType=*/ESIMem::I64,
                                       /*writeType=*/ESIMem::I3, TBackend>;
  using Tread0Ptr = std::shared_ptr<Tread0>;

  using Tloopback0 = ESICPP::ReadWritePort<
      /*readType=*/ESIMem::Struct16871797234873963366,
      /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>;
  using Tloopback0Ptr = std::shared_ptr<Tloopback0>;

  using Twrite0 = ESICPP::WritePort<
      /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>;
  using Twrite0Ptr = std::shared_ptr<Twrite0>;

  MemComms(Tread0Ptr read0, Tloopback0Ptr loopback0, Twrite0Ptr write0)
      : ESICPP::Module<TBackend>({read0, loopback0, write0}), read0(read0),
        loopback0(loopback0), write0(write0) {}

  std::shared_ptr<Tread0> read0;
  std::shared_ptr<Tloopback0> loopback0;
  std::shared_ptr<Twrite0> write0;
};

template <typename TBackend>
class DeclareRandomAccessMemory : public ESICPP::Module<TBackend> {
  using Port = ESICPP::Port<TBackend>;

public:
  // Port type declarations
  using Tread0 = ESICPP::ReadWritePort</*readType=*/ESIMem::I3,
                                       /*writeType=*/ESIMem::I64, TBackend>;
  using Tread0Ptr = std::shared_ptr<Tread0>;

  using Tread1 = ESICPP::ReadWritePort</*readType=*/ESIMem::I3,
                                       /*writeType=*/ESIMem::I64, TBackend>;
  using Tread1Ptr = std::shared_ptr<Tread1>;

  using Twrite0 = ESICPP::ReadWritePort<
      /*readType=*/ESIMem::Struct16871797234873963366,
      /*writeType=*/ESIMem::I1, TBackend>;
  using Twrite0Ptr = std::shared_ptr<Twrite0>;

  using Twrite1 = ESICPP::ReadWritePort<
      /*readType=*/ESIMem::Struct16871797234873963366,
      /*writeType=*/ESIMem::I1, TBackend>;
  using Twrite1Ptr = std::shared_ptr<Twrite1>;

  DeclareRandomAccessMemory(Tread0Ptr read0, Tread1Ptr read1, Twrite0Ptr write0,
                            Twrite1Ptr write1)
      : ESICPP::Module<TBackend>({read0, read1, write0, write1}), read0(read0),
        read1(read1), write0(write0), write1(write1) {}

  Tread0Ptr read0;
  Tread1Ptr read1;
  Twrite0Ptr write0;
  Twrite1Ptr write1;
  std::vector<std::shared_ptr<Port>> ports;
};

template <typename TBackend>
class Top {

public:
  Top(TBackend &backend) {

    { // declram initialization
      auto read0 = std::make_shared<
          ESICPP::ReadWritePort</*readType=*/ESIMem::I3,
                                /*writeType=*/ESIMem::I64, TBackend>>(
          std::vector<std::string>{"Mid", ""}, backend, "sv_mem");
      auto read1 = std::make_shared<
          ESICPP::ReadWritePort</*readType=*/ESIMem::I3,
                                /*writeType=*/ESIMem::I64, TBackend>>(
          std::vector<std::string>{""}, backend, "sv_mem");

      auto write0 = std::make_shared<ESICPP::ReadWritePort<
          /*readType=*/ESIMem::Struct16871797234873963366,
          /*writeType=*/ESIMem::I1, TBackend>>(
          std::vector<std::string>{"Mid", ""}, backend, "sv_mem");
      auto write1 = std::make_shared<ESICPP::ReadWritePort<
          /*readType=*/ESIMem::Struct16871797234873963366,
          /*writeType=*/ESIMem::I1, TBackend>>(std::vector<std::string>{""},
                                               backend, "sv_mem");

      declram = std::make_unique<DeclareRandomAccessMemory<TBackend>>(
          read0, read1, write0, write1);
    };

    {

      // memComms initialization
      auto read0 = std::make_shared<
          ESICPP::ReadWritePort</*readType=*/ESIMem::I64,
                                /*writeType=*/ESIMem::I3, TBackend>>(
          std::vector<std::string>{"read"}, backend, "cosim");

      auto loopback0 = std::make_shared<ESICPP::ReadWritePort<
          /*readType=*/ESIMem::Struct16871797234873963366,
          /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>>(
          std::vector<std::string>{"loopback"}, backend, "cosim");

      auto write0 = std::make_shared<ESICPP::WritePort<
          /*writeType=*/ESIMem::Struct16871797234873963366, TBackend>>(
          std::vector<std::string>{"write"}, backend, "cosim");
      memComms = std::make_unique<MemComms<TBackend>>(read0, loopback0, write0);
    };

  }; // namespace ESIMem

  std::unique_ptr<DeclareRandomAccessMemory<TBackend>> declram;
  std::unique_ptr<MemComms<TBackend>> memComms;
};

} // namespace ESIMem

namespace esi_test {
// Test namespace - this is all user-written code

template <typename TBackend>
int runTest(TBackend &backend) {
  // Connect the ESI system to the provided backend.
  ESIMem::Top top(backend);

  auto write_cmd = ESIMem::Struct16871797234873963366{.address = 2, .data = 42};

  auto loopback_result = (*top.memComms->loopback0)(write_cmd);
  if (loopback_result != write_cmd)
    return 1;

  auto read_result = (*top.memComms->read0)(2);
  if (read_result != ESIMem::I64(0))
    return 1;
  read_result = (*top.memComms->read0)(3);
  if (read_result != ESIMem::I64(0))
    return 1;

  (*top.memComms->write0)(write_cmd);
  read_result = (*top.memComms->read0)(2);
  if (read_result != ESIMem::I64(42))
    return 1;
  read_result = (*top.memComms->read0)(3);
  if (read_result != ESIMem::I64(42))
    return 1;

  return 0;
}

int run_cosim_test(const std::string &host, unsigned port) {
  // Run test with cosimulation backend.
  esi_cosim::Cosim cosim(host, port);
  return runTest(cosim);
}

} // namespace esi_test

int main(int argc, char **argv) {
  std::string rpchostport;
  if (argc != 2) {
    // Schema not currently used but required by the ESI cosim tester
    std::cerr << "usage: esi_ram_test configfile" << std::endl;
    return 1;
  }

  auto configFile = argv[1];

  // Parse the config file. It contains a line "port : ${port}"
  std::ifstream config(configFile);
  std::string line;
  while (std::getline(config, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos)
      continue;
    auto key = line.substr(0, colon);
    auto value = line.substr(colon + 1);
    if (key == "port") {
      rpchostport = "localhost:" + value;
      break;
    }
  }

  if (rpchostport.empty()) {
    std::cerr << "Could not find port in config file" << std::endl;
    return 1;
  }

  auto colon = rpchostport.find(':');
  auto host = rpchostport.substr(0, colon);
  auto port = stoi(rpchostport.substr(colon + 1));

  return esi_test::run_cosim_test(host, port);
}
