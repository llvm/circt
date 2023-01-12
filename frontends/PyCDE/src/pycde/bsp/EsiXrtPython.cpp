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
};

} // namespace

PYBIND11_MODULE(esiXrtPython, m) {
  py::class_<Accelerator>(m, "Accelerator")
      .def(py::init<const std::string &, const std::string &>());
}
