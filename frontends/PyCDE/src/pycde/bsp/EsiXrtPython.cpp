#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

// pybind11 includes
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_ip.h"
#include "experimental/xrt_xclbin.h"

#define IP_START_RD (1 << 0)
#define IP_DONE_RD (1 << 1)
#define IP_START_WR (1 << 4)
#define IP_DONE_WR (1 << 5)
#define CSR_OFFSET 0x0
#define ARG_OFFSET 0x10

// We don't want to clutter up the symbol space any more than necessary, so use
// an anonymous namespace.
namespace {
class Accelerator {
  xrt::device m_device;
  xrt::ip m_ip;

public:
  Accelerator(const std::string &xclbin_path) {
    const std::string kernel_name = "esi_bsp";
    m_device = xrt::device(0);
    auto uuid = m_device.load_xclbin(xclbin_path);
    m_ip = xrt::ip(m_device, uuid, kernel_name);
  }

  void sendMsg(uint32_t offset, uint32_t bitCount, py::int_ rawData) {
    uint32_t regCount = (bitCount + 31) / 32;
    for (uint32_t regNum = 0; regNum < regCount; ++regNum) {
      uint32_t data = (rawData >> (regNum * 32));
      m_ip.write_register(offset + regNum, data);
    }
  }

  py::object recvMsg(uint32_t offset, uint32_t bitCount) { return py::none(); }
};

} // namespace

PYBIND11_MODULE(esiXrtPython, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const std::string &>())
      .def("send_msg", &Accelerator::sendMsg)
      .def("recv_msg", &Accelerator::recvMsg);
}
