// RUN: circt-opt %s --verify-diagnostics --split-input-file

!bundleType = !esi.channel<i8>

hw.module @Receiver(in %foo: !bundleType, out data: !esi.channel<i8>) {
  %ack = esi.null : !esi.channel<none>
  // expected-error @+1 {{custom op 'esi.bundle.unpack' invalid kind of type specified}}
  %data = esi.bundle.unpack %ack from %foo : !bundleType
  hw.output %data : !esi.channel<i8>
}

// -----
