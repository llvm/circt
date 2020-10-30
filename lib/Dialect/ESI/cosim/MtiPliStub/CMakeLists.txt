set(SOURCES ${CAPNP_SRCS} DummySvDpi.cpp )
add_library(MtiPli SHARED ${SOURCES})
target_include_directories(MtiPli PRIVATE ${CIRCT_INCLUDE_DIR})
