//===-- circt-c/Dialect/FIRRTL.h - C API for FIRRTL dialect -------*- C -*-===//
//
// This header declares the C interface for registering and accessing the
// FIRRTL dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_C_DIALECT_FIRRTL_H
#define CIRCT_C_DIALECT_FIRRTL_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

// NOLINTBEGIN(modernize-use-using)

//===----------------------------------------------------------------------===//
// Opaque type declarations.
//
// Types are exposed to C bindings as structs containing opaque pointers. They
// are not supposed to be inspected from C. This allows the underlying
// representation to change without affecting the API users. The use of structs
// instead of typedefs enables some type safety as structs are not implicitly
// convertible to each other.
//
// Instances of these types may or may not own the underlying object. The
// ownership semantics is defined by how an instance of the type was obtained.
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(FirrtlContext, void);

#undef DEFINE_C_API_STRUCT

typedef MlirStringRef FirrtlStringRef;

/// Constructs a string reference from the pointer and length. The pointer need
/// not reference to a null-terminated string.
#define firrtlCreateStringRef mlirStringRefCreate

/// Constructs a string reference from a null-terminated C string. Prefer
/// `firrtlCreateStringRef` if the length of the string is known.
#define firrtlCreateStringRefFromCString mlirStringRefCreateFromCString

/// Returns true if two string references are equal, false otherwise.
#define firrtlStringRefEqual mlirStringRefEqual

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FIRRTL, firrtl);

MLIR_CAPI_EXPORTED FirrtlContext firrtlCreateContext(void);

MLIR_CAPI_EXPORTED void firrtlDestroyContext(FirrtlContext ctx);

typedef void (*FirrtlErrorHandler)(FirrtlStringRef message, void *userData);

MLIR_CAPI_EXPORTED void firrtlSetErrorHandler(FirrtlContext ctx,
                                              FirrtlErrorHandler handler,
                                              void *userData);

MLIR_CAPI_EXPORTED void firrtlVisitCircuit(FirrtlContext ctx,
                                           FirrtlStringRef name);

MLIR_CAPI_EXPORTED void firrtlVisitModule(FirrtlContext ctx,
                                          FirrtlStringRef name);

MLIR_CAPI_EXPORTED void firrtlVisitExtModule(FirrtlContext ctx,
                                             FirrtlStringRef name,
                                             FirrtlStringRef defName);

typedef enum FirrtlPortDirection {
  FIRRTL_PORT_DIRECTION_INPUT,
  FIRRTL_PORT_DIRECTION_OUTPUT,
} FirrtlPortDirection;

typedef enum FirrtlTypeKind {
  FIRRTL_TYPE_KIND_UINT = 0,
  FIRRTL_TYPE_KIND_SINT = 1,
  // FIRRTL_TYPE_KIND_FIXED = 2,    // Unsupported
  // FIRRTL_TYPE_KIND_INTERVAL = 3, // Unsupported
  FIRRTL_TYPE_KIND_CLOCK = 4,
  FIRRTL_TYPE_KIND_RESET,
  FIRRTL_TYPE_KIND_ASYNC_RESET,
  FIRRTL_TYPE_KIND_ANALOG,
  FIRRTL_TYPE_KIND_VECTOR,
  FIRRTL_TYPE_KIND_BUNDLE,
} FirrtlTypeKind;

typedef struct FirrtlTypeUInt {
  int32_t width;
} FirrtlTypeUInt;

typedef struct FirrtlTypeSInt {
  int32_t width;
} FirrtlTypeSInt;

typedef struct FirrtlTypeClock {
  // No fields
} FirrtlTypeClock;

typedef struct FirrtlTypeReset {
  // No fields
} FirrtlTypeReset;

typedef struct FirrtlTypeAsyncReset {
  // No fields
} FirrtlTypeAsyncReset;

typedef struct FirrtlTypeAnalog {
  int32_t width;
} FirrtlTypeAnalog;

typedef struct FirrtlType FirrtlType;

typedef struct FirrtlTypeVector {
  FirrtlType *type;
  size_t count;
} FirrtlTypeVector;

typedef struct FirrtlTypeBundleField {
  bool flip;
  FirrtlStringRef name;
  FirrtlType *type;
} FirrtlTypeBundleField;

typedef struct FirrtlTypeBundle {
  FirrtlTypeBundleField *fields;
  size_t count;
} FirrtlTypeBundle;

typedef union FirrtlTypeUnion {
  FirrtlTypeUInt uint;
  FirrtlTypeSInt sint;
  FirrtlTypeClock clock;
  FirrtlTypeReset reset;
  FirrtlTypeAsyncReset asyncReset;
  FirrtlTypeAnalog analog;
  FirrtlTypeVector vector;
  FirrtlTypeBundle bundle;
} FirrtlTypeUnion;

typedef struct FirrtlType {
  FirrtlTypeKind kind;
  FirrtlTypeUnion u;
} FirrtlType;

MLIR_CAPI_EXPORTED void firrtlVisitPort(FirrtlContext ctx, FirrtlStringRef name,
                                        FirrtlPortDirection direction,
                                        const FirrtlType *type);

MLIR_CAPI_EXPORTED FirrtlStringRef firrtlExportFirrtl(FirrtlContext ctx);

MLIR_CAPI_EXPORTED void firrtlDestroyString(FirrtlContext ctx,
                                            FirrtlStringRef string);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
