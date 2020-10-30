// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "circt/Dialect/ESI/cosim/EndPoint.h"
#include <iostream>

using namespace std;

EndPointRegistry::~EndPointRegistry() {
  Lock g(_M);
  EndPoints.clear();
}

void EndPointRegistry::RegisterEndPoint(int ep_id, long long esi_type_id,
                                        int type_size) {
  Lock g(_M);
  if (EndPoints.find(ep_id) != EndPoints.end()) {
    throw runtime_error("Endpoint ID already exists!");
  }
  EndPoints[ep_id] = make_unique<EndPoint>(esi_type_id, type_size);
}
