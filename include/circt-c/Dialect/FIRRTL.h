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

typedef enum FirrtlParameterKind {
  FIRRTL_PARAMETER_KIND_INT,
  FIRRTL_PARAMETER_KIND_DOUBLE,
  FIRRTL_PARAMETER_KIND_STRING,
  FIRRTL_PARAMETER_KIND_RAW
} FirrtlParameterKind;

typedef struct FirrtlParameterInt {
  int64_t value;
} FirrtlParameterInt;

typedef struct FirrtlParameterDouble {
  double value;
} FirrtlParameterDouble;

typedef struct FirrtlParameterString {
  FirrtlStringRef value;
} FirrtlParameterString;

typedef struct FirrtlParameterRaw {
  FirrtlStringRef value;
} FirrtlParameterRaw;

typedef union FirrtlParameterUnion {
  FirrtlParameterInt int_;
  FirrtlParameterDouble double_;
  FirrtlParameterString string;
  FirrtlParameterRaw raw;
} FirrtlParameterUnion;

typedef struct FirrtlParameter {
  FirrtlParameterKind kind;
  FirrtlParameterUnion u;
} FirrtlParameter;

MLIR_CAPI_EXPORTED void firrtlVisitParameter(FirrtlContext ctx,
                                             FirrtlStringRef name,
                                             const FirrtlParameter *param);

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

typedef enum FirrtlReadUnderWrite {
  FIRRTL_READ_UNDER_WRITE_UNDEFINED,
  FIRRTL_READ_UNDER_WRITE_OLD,
  FIRRTL_READ_UNDER_WRITE_NEW,
} FirrtlReadUnderWrite;

typedef enum FirrtlPrimOp {
  FIRRTL_PRIM_OP_VALIDIF,
  FIRRTL_PRIM_OP_ADD,
  FIRRTL_PRIM_OP_SUB,
  FIRRTL_PRIM_OP_TAIL,
  FIRRTL_PRIM_OP_HEAD,
  FIRRTL_PRIM_OP_MUL,
  FIRRTL_PRIM_OP_DIVIDE,
  FIRRTL_PRIM_OP_REM,
  FIRRTL_PRIM_OP_SHIFT_LEFT,
  FIRRTL_PRIM_OP_SHIFT_RIGHT,
  FIRRTL_PRIM_OP_DYNAMIC_SHIFT_LEFT,
  FIRRTL_PRIM_OP_DYNAMIC_SHIFT_RIGHT,
  FIRRTL_PRIM_OP_BIT_AND,
  FIRRTL_PRIM_OP_BIT_OR,
  FIRRTL_PRIM_OP_BIT_XOR,
  FIRRTL_PRIM_OP_BIT_NOT,
  FIRRTL_PRIM_OP_CONCAT,
  FIRRTL_PRIM_OP_BITS_EXTRACT,
  FIRRTL_PRIM_OP_LESS,
  FIRRTL_PRIM_OP_LESS_EQ,
  FIRRTL_PRIM_OP_GREATER,
  FIRRTL_PRIM_OP_GREATER_EQ,
  FIRRTL_PRIM_OP_EQUAL,
  FIRRTL_PRIM_OP_PAD,
  FIRRTL_PRIM_OP_NOT_EQUAL,
  FIRRTL_PRIM_OP_NEG,
  FIRRTL_PRIM_OP_MULTIPLEX,
  FIRRTL_PRIM_OP_AND_REDUCE,
  FIRRTL_PRIM_OP_OR_REDUCE,
  FIRRTL_PRIM_OP_XOR_REDUCE,
  FIRRTL_PRIM_OP_CONVERT,
  FIRRTL_PRIM_OP_AS_UINT,
  FIRRTL_PRIM_OP_AS_SINT,
  // FIRRTL_PRIM_OP_AS_FIXED_POINT, // Unsupported
  // FIRRTL_PRIM_OP_AS_INTERVAL,    // Unsupported
  FIRRTL_PRIM_OP_AS_CLOCK,
  FIRRTL_PRIM_OP_AS_ASYNC_RESET,
} FirrtlPrimOp;

typedef enum FirrtlExprKind {
  FIRRTL_EXPR_KIND_UINT,
  FIRRTL_EXPR_KIND_SINT,
  FIRRTL_EXPR_KIND_REF,
  FIRRTL_EXPR_KIND_PRIM,
} FirrtlExprKind;

typedef struct FirrtlExprUInt {
  uint64_t value;
  int32_t width;
} FirrtlExprUInt;

typedef struct FirrtlExprSInt {
  int64_t value;
  int32_t width;
} FirrtlExprSInt;

typedef struct FirrtlExprRef {
  FirrtlStringRef value;
} FirrtlExprRef;

typedef struct FirrtlPrim FirrtlPrim;

typedef struct FirrtlExprPrim {
  FirrtlPrim *value;
} FirrtlExprPrim;

typedef union FirrtlExprUnion {
  FirrtlExprUInt uint;
  FirrtlExprSInt sint;
  FirrtlExprRef ref;
  FirrtlExprPrim prim;
} FirrtlExprUnion;

typedef struct FirrtlExpr {
  FirrtlExprKind kind;
  FirrtlExprUnion u;
} FirrtlExpr;

typedef struct FirrtlPrimArgIntLit {
  int64_t value;
} FirrtlPrimArgIntLit;

typedef struct FirrtlPrimArgExpr {
  FirrtlExpr value;
} FirrtlPrimArgExpr;

typedef enum FirrtlPrimArgKind {
  FIRRTL_PRIM_ARG_KIND_INT_LIT,
  FIRRTL_PRIM_ARG_KIND_EXPR,
} FirrtlPrimArgKind;

typedef union FirrtlPrimArgUnion {
  FirrtlPrimArgIntLit intLit;
  FirrtlPrimArgExpr expr;
} FirrtlPrimArgUnion;

typedef struct FirrtlPrimArg {
  FirrtlPrimArgKind kind;
  FirrtlPrimArgUnion u;
} FirrtlPrimArg;

typedef struct FirrtlPrim {
  FirrtlPrimOp op;
  FirrtlPrimArg *args;
  size_t argsCount;
} FirrtlPrim;

typedef enum FirrtlStatementKind {
  FIRRTL_STATEMENT_KIND_ATTACH,
  FIRRTL_STATEMENT_KIND_SEQ_MEMORY,
  FIRRTL_STATEMENT_KIND_NODE,
  FIRRTL_STATEMENT_KIND_WIRE,
} FirrtlStatementKind;

typedef struct FirrtlStatementAttachOperand {
  FirrtlExpr expr;
} FirrtlStatementAttachOperand;

typedef struct FirrtlStatementAttach {
  FirrtlStatementAttachOperand *operands;
  size_t count;
} FirrtlStatementAttach;

typedef struct FirrtlStatementSeqMemory {
  FirrtlStringRef name;
  FirrtlType type;
  FirrtlReadUnderWrite readUnderWrite; // defaults to `FIRRTL_RUW_UNDEFINED`
} FirrtlStatementSeqMemory;

typedef struct FirrtlStatementNode {
  FirrtlStringRef name;
  FirrtlExpr expr;
} FirrtlStatementNode;

typedef struct FirrtlStatementWire {
  FirrtlStringRef name;
  FirrtlType type;
} FirrtlStatementWire;

typedef union FirrtlStatementUnion {
  FirrtlStatementAttach attach;
  FirrtlStatementSeqMemory seqMem;
  FirrtlStatementNode node;
  FirrtlStatementWire wire;
} FirrtlStatementUnion;

typedef struct FirrtlStatement {
  FirrtlStatementKind kind;
  FirrtlStatementUnion u;
} FirrtlStatement;

MLIR_CAPI_EXPORTED void firrtlVisitStatement(FirrtlContext ctx,
                                             const FirrtlStatement *stmt);

MLIR_CAPI_EXPORTED FirrtlStringRef firrtlExportFirrtl(FirrtlContext ctx);

MLIR_CAPI_EXPORTED void firrtlDestroyString(FirrtlContext ctx,
                                            FirrtlStringRef string);

// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
}
#endif

#endif // CIRCT_C_DIALECT_FIRRTL_H
