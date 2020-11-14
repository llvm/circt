//===- DpiEntryPoints.cpp - ESI cosim DPI calls -----------------*- C++ -*-===//
//
// Cosim DPI function implementations. Mostly C-C++ gaskets to the C++
// RpcServer.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/cosim/Server.h"
#include "circt/Dialect/ESI/cosim/dpi.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>

using namespace circt::esi::cosim;

static RpcServer *server = nullptr;

// ---- Helper functions ----

/// Get the TCP port on which to listen. Defaults to 0xECD (ESI Cosim DPI), 3789
/// in decimal.
int getPort() {
  const char *portEnv = getenv("COSIM_PORT");
  if (portEnv == nullptr) {
    printf("[COSIM] RPC server port not found. Defaulting to 3789.\n");
    return 0xECD;
  }
  printf("[COSIM] Opening RPC server on port %s\n", portEnv);
  return std::strtoull(portEnv, nullptr, 10);
}

/// Check that an array is an array of bytes and has some size.
// NOLINTNEXTLINE(misc-misplaced-const)
static int validateSvOpenArray(const svOpenArrayHandle data,
                               int expectedElemSize) {
  if (svDimensions(data) != 1) {
    printf("DPI-C: ERROR passed array argument that doesn't have expected 1D "
           "dimensions\n");
    return -1;
  }
  if (svGetArrayPtr(data) == NULL) {
    printf("DPI-C: ERROR passed array argument that doesn't have C layout "
           "(ptr==NULL)\n");
    return -2;
  }
  int totalBytes = svSizeOfArray(data);
  if (totalBytes == 0) {
    printf("DPI-C: ERROR passed array argument that doesn't have C layout "
           "(total_bytes==0)\n");
    return -3;
  }
  int numElems = svSize(data, 1);
  int elemSize = numElems == 0 ? 0 : (totalBytes / numElems);
  if (numElems * expectedElemSize != totalBytes) {
    printf("DPI-C: ERROR: passed array argument that doesn't have expected "
           "element-size: expected=%d actual=%d\n",
           expectedElemSize, elemSize);
    return -4;
  }
  return 0;
}

// ---- DPI entry points ----

// Register simulated device endpoints.
// - return 0 on success, non-zero on failure (duplicate EP registered).
DPI int sv2cCosimserverEpRegister(int endpointId, long long sendTypeId,
                                  int sendTypeSize, long long recvTypeId,
                                  int recvTypeSize) {
  // Ensure the server has been constructed.
  sv2cCosimserverInit();
  try {
    // Then register with it.
    server->endpoints.registerEndpoint(endpointId, sendTypeId, sendTypeSize,
                                       recvTypeId, recvTypeSize);
    return 0;
  } catch (const std::runtime_error &rt) {
    fprintf(stderr, "%s\n", rt.what());
    return -1;
  }
}

// Attempt to recieve data from a client.
//   - Returns negative when call failed (e.g. EP not registered).
//   - If no message, return 0 with dataSize == 0.
//   - Assumes buffer is large enough to contain entire message. Fails if not
//     large enough. (In the future, will add support for getting the message
//     into a fixed-size buffer over multiple calls.)
DPI int sv2cCosimserverEpTryGet(unsigned int endpointId,
                                // NOLINTNEXTLINE(misc-misplaced-const)
                                const svOpenArrayHandle data,
                                unsigned int *dataSize) {
  if (server == nullptr)
    return -1;

  try {
    Endpoint::BlobPtr msg;
    // Poll for a message.
    if (server->endpoints[endpointId].getMessageToSim(msg)) {

      // Do the validation only if there's a message available. Since the
      // simulator is going to poll up to every tick and there's not going to be
      // a message most of the time, this is important for performance.

      if (validateSvOpenArray(data, sizeof(int8_t)) != 0) {
        printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array\n", __func__,
               __LINE__);
        return -2;
      }

      // Detect or verify size of buffer.
      if (*dataSize == ~0u) {
        *dataSize = svSizeOfArray(data);
      } else if (*dataSize > (unsigned)svSizeOfArray(data)) {
        printf("ERROR: DPI-func=%s line %d event=invalid-size (max %d)\n",
               __func__, __LINE__, (unsigned)svSizeOfArray(data));
        return -3;
      }
      // Verify it'll fit.
      if (msg->size() > *dataSize) {
        printf("ERROR: Message size too big to fit in RTL buffer\n");
        return -4;
      }

      // Copy the message data.
      size_t i;
      for (i = 0; i < msg->size(); ++i) {
        auto b = msg->at(i);
        *(char *)svGetArrElemPtr1(data, i) = b;
      }
      // Zero out the rest of the buffer.
      for (; i < (*dataSize); ++i) {
        *(char *)svGetArrElemPtr1(data, i) = 0;
      }
      // Set the output data size.
      *dataSize = msg->size();
      return 0;
    } else {
      // No message.
      *dataSize = 0;
      return 0;
    }
  } catch (const std::runtime_error &rt) {
    fprintf(stderr, "%s\n", rt.what());
    return -5;
  }
}

// Attempt to send data to a client.
// - return 0 on success, negative on failure (unregistered EP).
DPI int sv2cCosimserverEpTryPut(unsigned int endpointId,
                                // NOLINTNEXTLINE(misc-misplaced-const)
                                const svOpenArrayHandle data, int dataSize) {
  if (server == nullptr)
    return -1;

  if (validateSvOpenArray(data, sizeof(int8_t)) != 0) {
    printf("ERROR: DPI-func=%s line=%d event=invalid-sv-array\n", __func__,
           __LINE__);
    return -2;
  }

  // Detect or verify size.
  if (dataSize < 0) {
    dataSize = svSizeOfArray(data);
  } else if (dataSize > svSizeOfArray(data)) { // not enough data
    printf("ERROR: DPI-func=%s line %d event=invalid-size limit %d array %d\n",
           __func__, __LINE__, dataSize, svSizeOfArray(data));
    return -3;
  }

  try {
    Endpoint::BlobPtr blob = std::make_shared<Endpoint::Blob>(dataSize);
    // Copy the message data into 'blob'.
    for (long i = 0; i < dataSize; ++i) {
      blob->at(i) = *(char *)svGetArrElemPtr1(data, i);
    }
    // Queue the blob.
    server->endpoints[endpointId].pushMessageToClient(blob);
    return 0;
  } catch (const std::runtime_error &e) {
    fprintf(stderr, "%s\n", e.what());
    return -4;
  }
}

// Teardown cosimserver (disconnects from primary server port, stops connections
// from active clients).
DPI void sv2cCosimserverFini() {
  printf("[cosim] Tearing down RPC server.\n");
  if (server != nullptr) {
    server->stop();
    server = nullptr;
  }
}

// Start cosimserver (spawns server for RTL-initiated work, listens for
// connections from new SW-clients).
DPI int sv2cCosimserverInit() {
  if (server == nullptr) {
    printf("[cosim] Starting RPC server.\n");
    server = new RpcServer();
    server->run(getPort());
  }
  return 0;
}
