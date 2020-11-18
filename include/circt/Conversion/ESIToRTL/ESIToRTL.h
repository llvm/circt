//===- ESIToRtl.h - ESI dialect to RTL/SV dialects --------------*- C++ -*-===//
//
// Register the ESI conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_ESITORTL_H_
#define CIRCT_CONVERSION_ESITORTL_H_

namespace circt {
namespace esi {
void registerESIToRTLPasses();
}
} // namespace circt

#endif
