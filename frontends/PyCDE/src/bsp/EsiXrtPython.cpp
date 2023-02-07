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

#define VALID (1 << 0)
#define DONE (1 << 1)
#define DATA_WIDTH 32

// We don't want to clutter up the symbol space any more than necessary, so use
// an anonymous namespace.
namespace {
class Accelerator {
  xrt::device m_device;
  xrt::ip m_ip;

public:
  Accelerator(const std::string &xclbin_path, const std::string kernel_name) {
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
    fflush(stdout);
  }

  py::object recvMsg(uint32_t offset, uint32_t bitCount) {
    uint32_t ctrl = m_ip.read_register(offset);
    if ((ctrl & VALID) == 0)
      return py::none();
    size_t numRegs = (bitCount + DATA_WIDTH - 1) / DATA_WIDTH;
    py::int_ ret;
    for (uint32_t i = 1; i <= numRegs; ++i) {
      uint32_t num = offset + i * 4;
      uint32_t val = m_ip.read_register(num);
      ret = ret | (py::int_(val) << (i * DATA_WIDTH));
    }
    m_ip.write_register(offset, DONE);
    return ret;
  }
};

} // namespace

PYBIND11_MODULE(esiXrtPython, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const std::string &, const std::string &>())
      .def("send_msg", &Accelerator::sendMsg)
      .def("recv_msg", &Accelerator::recvMsg);
}
