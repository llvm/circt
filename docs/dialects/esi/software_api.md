[Back to table of contents](index.md#Table-of-contents)

# ESI Software APIs

*More on this to be written.*

Thanks to ESI's types, typed, design-dependent software APIs can be
automatically generated. Said APIs would be mostly independent of the
transport mechanism (PCIe, network, etc.) used to communicate with the
silicon. The same API could even drive a simulation of the ESI system.

Said APIs would need to ensure that the software is talking to the correct
hardware. There are several possible approaches:

1) Generate a hash which is contained in the API and the hardware then compare
at runtime. This has the downside of software not being able to communicate
with logically compatible hardware. (E.g. mismatched interfaces aren't used.)
2) Communicate via some self-describing message format (e.g. protobuf or
bond). This has the downside of encoding/decoding overhead.
3) Auto-generate self-describing hardware for dynamic discovery of
functionality. This would allow software to be built without a static API
(e.g. a generic poke/prod script). The downside is larger hardware area.

ESI will initially support option 2, using
[Capn'Proto](https://capnproto.org/). Capnp uses a schema-based approach to
avoid the overhead self-description in the messages. It uses a data layout
very similar to C for non-pointer types.
