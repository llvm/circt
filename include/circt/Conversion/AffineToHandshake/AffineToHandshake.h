//===- AffineToHandshake.h --------------------------------------*- C++ -*-===//
//
// This file declares the registration interface for the affine-to-handshake
// conversion pass.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_
#define CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_

namespace circt {
namespace handshake {
void registerAffineToHandshakePasses();
}
} // namespace circt

#endif // CIRCT_CONVERSION_AFFINETOHANDSHAKE_H_
