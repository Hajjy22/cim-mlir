//===- Interpreter.cpp - Execute cim IR against cimrt ----------*- C++ -*-===//

#include "cim/Interpreter/Interpreter.h"

#include "cim/Dialect/CIMOps.h"
#include "cimrt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

#include <cstring>
#include <memory>
#include <vector>

using namespace mlir;
using namespace mlir::cim;

namespace {

//===----------------------------------------------------------------------===//
// Memory model
//===----------------------------------------------------------------------===//

/// One real buffer. Backed either by host memory or by a cimrt_buffer,
/// decided by the memref's #cim.space attribute.
struct Allocation {
  std::vector<uint8_t> host;
  cimrt_buffer *device = nullptr;
  bool onDevice = false;
  size_t sizeBytes = 0;

  ~Allocation() {
    if (device)
      cimrt_free(device);
  }
  Allocation() = default;
  Allocation(const Allocation &) = delete;
  Allocation &operator=(const Allocation &) = delete;
};

/// A (possibly strided) view of one allocation. Offsets and strides are
/// read off the memref type rather than recomputed -- MLIR's
/// SubViewOp::inferResultType has already composed them into
/// strided<[...], offset: N>, absolute with respect to a root allocation
/// that always has identity layout here.
struct MemRefValue {
  std::shared_ptr<Allocation> alloc;
  int64_t offsetElems = 0;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> strides;
  unsigned elemBytes = 1;

  int64_t numElements() const {
    int64_t n = 1;
    for (int64_t s : sizes)
      n *= s;
    return n;
  }
};

struct Resident {
  cimrt_tile_id tile = 0;
  int64_t rows = 0;
  int64_t cols = 0;
};

/// True when this memref lives in a cim address space other than host.
bool isDeviceSpace(MemRefType type) {
  auto space = llvm::dyn_cast_or_null<SpaceAttr>(type.getMemorySpace());
  return space && space.getKind() != SpaceKind::host;
}

//===----------------------------------------------------------------------===//
// Interpreter
//===----------------------------------------------------------------------===//

class Interpreter {
public:
  Interpreter(const InterpreterOptions &opts, llvm::raw_ostream &stream)
      : options(opts), out(stream) {}

  ~Interpreter() {
    for (auto &entry : devices)
      cimrt_close(entry.second);
  }

  LogicalResult run(ModuleOp module, StringRef entryName);

private:
  LogicalResult execute(Operation *op);
  LogicalResult executeBlock(Block &block);

  // --- helpers ---
  FailureOr<MemRefValue> viewFromType(MemRefType type,
                                       std::shared_ptr<Allocation> base);
  std::shared_ptr<Allocation> makeAllocation(MemRefType type, size_t bytes,
                                              cimrt_device *dev);
  LogicalResult readElement(const MemRefValue &v, ArrayRef<int64_t> idx,
                            void *dst);
  LogicalResult writeElement(const MemRefValue &v, ArrayRef<int64_t> idx,
                             const void *src);
  /// Flatten a (possibly strided) view into contiguous bytes.
  LogicalResult gather(const MemRefValue &v, std::vector<uint8_t> &outBytes);
  LogicalResult scatter(const MemRefValue &v, const std::vector<uint8_t> &in);

  FailureOr<cimrt_device *> deviceFor(Operation *op, StringRef target);

  /// A previously bound scalar index/integer value, or failure if `v` was
  /// never computed by an op this interpreter models (arith.constant, or a
  /// bound scf.for induction variable). Never guesses -- see runSubView.
  FailureOr<int64_t> evalIndex(Value v) const;
  /// Resolve one offset/size/stride entry of a subview: a compile-time
  /// constant if the op encoded one, otherwise a lookup through evalIndex.
  FailureOr<int64_t> resolveOffsetLike(OpFoldResult ofr) const;

  // --- op handlers ---
  LogicalResult runGetGlobal(memref::GetGlobalOp op);
  LogicalResult runAlloc(Operation *op, MemRefType type);
  LogicalResult runSubView(memref::SubViewOp op);
  LogicalResult runMemRefCopy(memref::CopyOp op);
  LogicalResult runCall(func::CallOp op);
  LogicalResult runDeviceOpen(DeviceOpenOp op);
  LogicalResult runTileAlloc(TileAllocOp op);
  LogicalResult runProgram(ProgramOp op);
  LogicalResult runMvm(MvmOp op);
  LogicalResult runReducePartial(ReducePartialOp op);
  LogicalResult runCimCopy(CopyOp op);
  LogicalResult runArithConstant(arith::ConstantOp op);
  LogicalResult runScfFor(scf::ForOp op);

  const InterpreterOptions &options;
  llvm::raw_ostream &out;

  ModuleOp module;
  llvm::DenseMap<Value, MemRefValue> memrefs;
  llvm::DenseMap<Value, cimrt_device *> deviceValues;
  llvm::DenseMap<Value, cimrt_tile_id> tileValues;
  llvm::DenseMap<Value, Resident> residentValues;
  llvm::StringMap<cimrt_device *> devices;
  /// Scalar index/integer SSA values the interpreter has computed: constants
  /// and scf.for induction variables. Nothing here is inferred by folding
  /// through arbitrary arithmetic -- see the note on runScfFor's scope.
  llvm::DenseMap<Value, int64_t> indexValues;
};

/// Visit every index of a shape in row-major order.
void forEachIndex(ArrayRef<int64_t> sizes,
                  llvm::function_ref<void(ArrayRef<int64_t>)> fn) {
  SmallVector<int64_t> idx(sizes.size(), 0);
  if (sizes.empty()) {
    fn(idx);
    return;
  }
  while (true) {
    fn(idx);
    int64_t dim = static_cast<int64_t>(sizes.size()) - 1;
    while (dim >= 0) {
      if (++idx[dim] < sizes[dim])
        break;
      idx[dim] = 0;
      --dim;
    }
    if (dim < 0)
      return;
  }
}

FailureOr<MemRefValue>
Interpreter::viewFromType(MemRefType type, std::shared_ptr<Allocation> base) {
  SmallVector<int64_t> strides;
  int64_t offset = 0;
  // Free function in MLIR 18; becomes MemRefType::getStridesAndOffset in
  // MLIR >= 20.
  if (failed(getStridesAndOffset(type, strides, offset)))
    return failure();
  if (!type.hasStaticShape() || ShapedType::isDynamic(offset))
    return failure();
  for (int64_t s : strides)
    if (ShapedType::isDynamic(s))
      return failure();

  MemRefValue v;
  v.alloc = std::move(base);
  v.offsetElems = offset;
  v.sizes = llvm::to_vector(type.getShape());
  v.strides = strides;
  v.elemBytes = type.getElementTypeBitWidth() / 8;
  return v;
}

std::shared_ptr<Allocation> Interpreter::makeAllocation(MemRefType type,
                                                         size_t bytes,
                                                         cimrt_device *dev) {
  auto alloc = std::make_shared<Allocation>();
  alloc->sizeBytes = bytes;
  if (isDeviceSpace(type) && dev) {
    alloc->onDevice = true;
    auto space = llvm::cast<SpaceAttr>(type.getMemorySpace());
    const cimrt_space rtSpace = space.getKind() == SpaceKind::insitu
                                    ? CIMRT_SPACE_INSITU
                                    : CIMRT_SPACE_NEAR;
    if (cimrt_alloc(dev, bytes, rtSpace, &alloc->device) != CIMRT_OK)
      return nullptr;
  } else {
    alloc->host.assign(bytes, 0);
  }
  return alloc;
}

LogicalResult Interpreter::readElement(const MemRefValue &v,
                                       ArrayRef<int64_t> idx, void *dst) {
  int64_t elem = v.offsetElems;
  for (size_t d = 0; d < idx.size(); ++d)
    elem += idx[d] * v.strides[d];
  const size_t byteOffset = static_cast<size_t>(elem) * v.elemBytes;

  if (v.alloc->onDevice)
    return success(cimrt_read(v.alloc->device, byteOffset, dst, v.elemBytes) ==
                   CIMRT_OK);
  if (byteOffset + v.elemBytes > v.alloc->host.size())
    return failure();
  std::memcpy(dst, v.alloc->host.data() + byteOffset, v.elemBytes);
  return success();
}

LogicalResult Interpreter::writeElement(const MemRefValue &v,
                                        ArrayRef<int64_t> idx,
                                        const void *src) {
  int64_t elem = v.offsetElems;
  for (size_t d = 0; d < idx.size(); ++d)
    elem += idx[d] * v.strides[d];
  const size_t byteOffset = static_cast<size_t>(elem) * v.elemBytes;

  if (v.alloc->onDevice)
    return success(cimrt_write(v.alloc->device, byteOffset, src, v.elemBytes) ==
                   CIMRT_OK);
  if (byteOffset + v.elemBytes > v.alloc->host.size())
    return failure();
  std::memcpy(v.alloc->host.data() + byteOffset, src, v.elemBytes);
  return success();
}

LogicalResult Interpreter::gather(const MemRefValue &v,
                                  std::vector<uint8_t> &outBytes) {
  outBytes.assign(static_cast<size_t>(v.numElements()) * v.elemBytes, 0);
  size_t cursor = 0;
  LogicalResult status = success();
  forEachIndex(v.sizes, [&](ArrayRef<int64_t> idx) {
    if (failed(status))
      return;
    if (failed(readElement(v, idx, outBytes.data() + cursor)))
      status = failure();
    cursor += v.elemBytes;
  });
  return status;
}

LogicalResult Interpreter::scatter(const MemRefValue &v,
                                   const std::vector<uint8_t> &in) {
  size_t cursor = 0;
  LogicalResult status = success();
  forEachIndex(v.sizes, [&](ArrayRef<int64_t> idx) {
    if (failed(status))
      return;
    if (failed(writeElement(v, idx, in.data() + cursor)))
      status = failure();
    cursor += v.elemBytes;
  });
  return status;
}

FailureOr<cimrt_device *> Interpreter::deviceFor(Operation *op,
                                                  StringRef target) {
  auto it = devices.find(target);
  if (it != devices.end())
    return it->second;

  // The target attribute is a bare name; the caller supplies the path.
  const std::string path =
      options.targetYAMLPath.empty() ? target.str() : options.targetYAMLPath;
  cimrt_device *dev = nullptr;
  const cimrt_status status = cimrt_open(path.c_str(), &dev);
  if (status != CIMRT_OK) {
    op->emitError("cimrt_open(\"") << path << "\") failed: "
                                   << cimrt_status_string(status);
    return failure();
  }
  devices[target] = dev;
  return dev;
}

FailureOr<int64_t> Interpreter::evalIndex(Value v) const {
  auto it = indexValues.find(v);
  if (it == indexValues.end())
    return failure();
  return it->second;
}

FailureOr<int64_t> Interpreter::resolveOffsetLike(OpFoldResult ofr) const {
  if (std::optional<int64_t> constant = getConstantIntValue(ofr))
    return *constant;
  // Not a compile-time constant: the only other thing an offset/size/stride
  // operand can be is a Value, and the only Values this interpreter binds
  // are constants and loop induction variables (see runScfFor). Anything
  // else -- a value computed by arithmetic this interpreter does not model,
  // or a genuinely runtime-dependent quantity -- is a hard error, not a
  // guess: guessing an offset would silently read or write the wrong bytes.
  return evalIndex(llvm::cast<Value>(ofr));
}

//===----------------------------------------------------------------------===//
// Op handlers
//===----------------------------------------------------------------------===//

LogicalResult Interpreter::runGetGlobal(memref::GetGlobalOp op) {
  auto type = llvm::cast<MemRefType>(op.getType());
  auto global = module.lookupSymbol<memref::GlobalOp>(op.getNameAttr());
  if (!global)
    return op.emitError("no such memref.global: ") << op.getName();

  auto initAttr = global.getInitialValue();
  if (!initAttr)
    return op.emitError(
        "memref.global has no initial value; the interpreter cannot invent "
        "one, and treating it as zero would silently change the result");
  auto elements = llvm::dyn_cast<DenseIntElementsAttr>(*initAttr);
  if (!elements)
    return op.emitError("only dense integer memref.global initializers are "
                        "supported");

  const unsigned elemBytes = type.getElementTypeBitWidth() / 8;
  auto alloc = makeAllocation(type, static_cast<size_t>(type.getNumElements()) *
                                        elemBytes,
                              /*dev=*/nullptr);
  if (!alloc)
    return op.emitError("failed to allocate for memref.global");

  size_t cursor = 0;
  for (const APInt &value : elements.getValues<APInt>()) {
    const uint64_t raw = value.getZExtValue();
    for (unsigned b = 0; b < elemBytes; ++b)
      alloc->host[cursor + b] = static_cast<uint8_t>((raw >> (8 * b)) & 0xFF);
    cursor += elemBytes;
  }

  auto view = viewFromType(type, alloc);
  if (failed(view))
    return op.emitError("unsupported memref layout on memref.global");
  memrefs[op.getResult()] = *view;
  return success();
}

LogicalResult Interpreter::runAlloc(Operation *op, MemRefType type) {
  if (!type.hasStaticShape())
    return op->emitError("dynamic allocations are not supported");
  const unsigned elemBytes = type.getElementTypeBitWidth() / 8;
  auto alloc = makeAllocation(
      type, static_cast<size_t>(type.getNumElements()) * elemBytes, nullptr);
  if (!alloc)
    return op->emitError("allocation failed");
  auto view = viewFromType(type, alloc);
  if (failed(view))
    return op->emitError("unsupported memref layout on allocation");
  memrefs[op->getResult(0)] = *view;
  return success();
}

LogicalResult Interpreter::runSubView(memref::SubViewOp op) {
  auto it = memrefs.find(op.getSource());
  if (it == memrefs.end())
    return op.emitError("subview of an unknown buffer");
  const MemRefValue &src = it->second;

  // A subview's offset/size/stride operand list always has one entry per
  // SOURCE dimension, regardless of how many dimensions the result drops --
  // that is the invariant this composition relies on.
  const SmallVector<OpFoldResult> mixedOffsets = op.getMixedOffsets();
  const SmallVector<OpFoldResult> mixedSizes = op.getMixedSizes();
  const SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();
  if (mixedOffsets.size() != src.strides.size() ||
      mixedSizes.size() != src.strides.size() ||
      mixedStrides.size() != src.strides.size())
    return op.emitError("cim interpreter: subview operand count does not "
                        "match its source's rank");

  // For an all-static subview this reproduces exactly what the old
  // type-only path computed (MLIR composes the same arithmetic at compile
  // time into the result type's strided<[...], offset: N>). What is new is
  // resolving an offset that is a bound index value -- typically an
  // enclosing scf.for's induction variable -- rather than refusing it. Any
  // offset this interpreter cannot resolve is still a hard error: see
  // resolveOffsetLike.
  int64_t offsetElems = src.offsetElems;
  SmallVector<int64_t> fullSizes(mixedSizes.size());
  SmallVector<int64_t> fullStrides(mixedStrides.size());
  for (size_t d = 0, e = mixedOffsets.size(); d < e; ++d) {
    FailureOr<int64_t> resolvedOffset = resolveOffsetLike(mixedOffsets[d]);
    FailureOr<int64_t> resolvedSize = resolveOffsetLike(mixedSizes[d]);
    FailureOr<int64_t> resolvedStride = resolveOffsetLike(mixedStrides[d]);
    if (failed(resolvedOffset) || failed(resolvedSize) ||
        failed(resolvedStride))
      return op.emitError(
          "memref.subview offset, size or stride is not a compile-time "
          "constant or a value the interpreter has bound (currently only "
          "arith.constant and scf.for induction variables)");
    offsetElems += *resolvedOffset * src.strides[d];
    fullSizes[d] = *resolvedSize;
    fullStrides[d] = src.strides[d] * *resolvedStride;
  }

  const llvm::SmallBitVector dropped = op.getDroppedDims();
  MemRefValue result;
  result.alloc = src.alloc;
  result.offsetElems = offsetElems;
  result.elemBytes = src.elemBytes;
  for (size_t d = 0, e = fullSizes.size(); d < e; ++d) {
    if (dropped.test(d))
      continue;
    result.sizes.push_back(fullSizes[d]);
    result.strides.push_back(fullStrides[d]);
  }
  memrefs[op.getResult()] = result;
  return success();
}

LogicalResult Interpreter::runMemRefCopy(memref::CopyOp op) {
  auto srcIt = memrefs.find(op.getSource());
  auto dstIt = memrefs.find(op.getTarget());
  if (srcIt == memrefs.end() || dstIt == memrefs.end())
    return op.emitError("memref.copy on an unknown buffer");

  std::vector<uint8_t> bytes;
  if (failed(gather(srcIt->second, bytes)))
    return op.emitError("memref.copy failed to read its source");
  if (failed(scatter(dstIt->second, bytes)))
    return op.emitError("memref.copy failed to write its target");
  return success();
}

LogicalResult Interpreter::runCall(func::CallOp op) {
  if (!op.getCallee().starts_with("cim_print_"))
    return op.emitError("the interpreter only models @cim_print_* calls, not ")
           << op.getCallee();
  if (op.getNumOperands() != 1)
    return op.emitError("@cim_print_* takes exactly one operand");

  auto it = memrefs.find(op.getOperand(0));
  if (it == memrefs.end())
    return op.emitError("printing an unknown buffer");
  const MemRefValue &v = it->second;

  std::vector<uint8_t> bytes;
  if (failed(gather(v, bytes)))
    return op.emitError("failed to read the buffer being printed");

  out << op.getCallee() << " shape=[";
  for (size_t i = 0; i < v.sizes.size(); ++i)
    out << (i ? "," : "") << v.sizes[i];
  out << "] data=[";
  const int64_t count = v.numElements();
  for (int64_t i = 0; i < count; ++i) {
    if (i)
      out << ",";
    if (v.elemBytes == 4) {
      int32_t value = 0;
      std::memcpy(&value, bytes.data() + i * 4, 4);
      out << value;
    } else {
      int8_t value = 0;
      std::memcpy(&value, bytes.data() + i, 1);
      out << static_cast<int>(value);
    }
  }
  out << "]\n";
  return success();
}

LogicalResult Interpreter::runDeviceOpen(DeviceOpenOp op) {
  auto dev = deviceFor(op, op.getTarget());
  if (failed(dev))
    return failure();
  deviceValues[op.getResult()] = *dev;
  return success();
}

LogicalResult Interpreter::runTileAlloc(TileAllocOp op) {
  auto it = deviceValues.find(op.getDevice());
  if (it == deviceValues.end())
    return op.emitError("cim.tile_alloc on an unopened device");

  cimrt_device_info info{};
  if (cimrt_query(it->second, &info) != CIMRT_OK)
    return op.emitError("cimrt_query failed");

  const uint64_t id = op.getId();
  if (id >= info.num_tiles)
    return op.emitError("tile id ")
           << id << " is outside the target's " << info.num_tiles << " tiles";

  tileValues[op.getResult()] = static_cast<cimrt_tile_id>(id);
  return success();
}

LogicalResult Interpreter::runProgram(ProgramOp op) {
  auto tileIt = tileValues.find(op.getTile());
  if (tileIt == tileValues.end())
    return op.emitError("cim.program on an unallocated tile");
  auto weightsIt = memrefs.find(op.getWeights());
  if (weightsIt == memrefs.end())
    return op.emitError("cim.program on an unknown weight buffer");

  auto deviceOp = op.getTile().getDefiningOp<TileAllocOp>();
  if (!deviceOp)
    return op.emitError("cim.program's tile did not come from cim.tile_alloc");
  auto devIt = deviceValues.find(deviceOp.getDevice());
  if (devIt == deviceValues.end())
    return op.emitError("cim.program on an unopened device");

  // The weight sub-matrix is a strided view of the host weight buffer, so
  // it must be gathered into a contiguous device buffer. That is not an
  // artifact of the interpreter: on real hardware the block genuinely has
  // to be moved.
  std::vector<uint8_t> bytes;
  if (failed(gather(weightsIt->second, bytes)))
    return op.emitError("failed to read the weight sub-matrix");

  cimrt_buffer *staging = nullptr;
  if (cimrt_alloc(devIt->second, bytes.size(), CIMRT_SPACE_INSITU, &staging) !=
      CIMRT_OK)
    return op.emitError("failed to allocate a staging buffer for the weights");
  const cimrt_status wrote =
      cimrt_write(staging, 0, bytes.data(), bytes.size());
  const cimrt_status programmed =
      wrote == CIMRT_OK
          ? cimrt_program(devIt->second, tileIt->second, staging)
          : wrote;
  cimrt_free(staging);
  if (programmed != CIMRT_OK)
    return op.emitError("cimrt_program failed: ")
           << cimrt_status_string(programmed);

  auto residentType = op.getResult().getType();
  residentValues[op.getResult()] = Resident{tileIt->second,
                                             residentType.getShape()[0],
                                             residentType.getShape()[1]};
  return success();
}

LogicalResult Interpreter::runMvm(MvmOp op) {
  auto resIt = residentValues.find(op.getWeights());
  if (resIt == residentValues.end())
    return op.emitError("cim.mvm against weights that were never programmed");
  auto actIt = memrefs.find(op.getActivations());
  if (actIt == memrefs.end())
    return op.emitError("cim.mvm on an unknown activation buffer");

  auto programOp = op.getWeights().getDefiningOp<ProgramOp>();
  auto tileAllocOp =
      programOp ? programOp.getTile().getDefiningOp<TileAllocOp>() : nullptr;
  if (!tileAllocOp)
    return op.emitError("could not trace cim.mvm back to its device");
  auto devIt = deviceValues.find(tileAllocOp.getDevice());
  if (devIt == deviceValues.end())
    return op.emitError("cim.mvm on an unopened device");
  cimrt_device *dev = devIt->second;

  std::vector<uint8_t> actBytes;
  if (failed(gather(actIt->second, actBytes)))
    return op.emitError("failed to read the activation slice");

  cimrt_buffer *actBuf = nullptr;
  if (cimrt_alloc(dev, actBytes.size(), CIMRT_SPACE_NEAR, &actBuf) != CIMRT_OK)
    return op.emitError("failed to allocate an activation buffer");
  if (cimrt_write(actBuf, 0, actBytes.data(), actBytes.size()) != CIMRT_OK) {
    cimrt_free(actBuf);
    return op.emitError("failed to stage activations");
  }

  auto resultType = llvm::cast<MemRefType>(op.getResult().getType());
  const unsigned elemBytes = resultType.getElementTypeBitWidth() / 8;
  const size_t resultBytes =
      static_cast<size_t>(resultType.getNumElements()) * elemBytes;

  auto alloc = makeAllocation(resultType, resultBytes, dev);
  if (!alloc) {
    cimrt_free(actBuf);
    return op.emitError("failed to allocate the mvm result");
  }

  // cim.mvm produces a fresh allocation rather than writing in place. The
  // accumulate flag chains into an existing result, which is what
  // cim.reduce_partial expresses at the IR level, so it is false here.
  cimrt_status status;
  if (alloc->onDevice) {
    status = cimrt_mvm(dev, resIt->second.tile, actBuf, alloc->device,
                       op.getAccumulate());
  } else {
    cimrt_buffer *tmp = nullptr;
    status = cimrt_alloc(dev, resultBytes, CIMRT_SPACE_NEAR, &tmp);
    if (status == CIMRT_OK) {
      status = cimrt_mvm(dev, resIt->second.tile, actBuf, tmp,
                         op.getAccumulate());
      if (status == CIMRT_OK)
        status = cimrt_read(tmp, 0, alloc->host.data(), resultBytes);
      cimrt_free(tmp);
    }
  }
  cimrt_free(actBuf);
  if (status != CIMRT_OK)
    return op.emitError("cimrt_mvm failed: ") << cimrt_status_string(status);

  auto view = viewFromType(resultType, alloc);
  if (failed(view))
    return op.emitError("unsupported layout on the mvm result");
  memrefs[op.getResult()] = *view;
  return success();
}

LogicalResult Interpreter::runReducePartial(ReducePartialOp op) {
  auto resultType = llvm::cast<MemRefType>(op.getResult().getType());
  const int64_t count = resultType.getNumElements();

  std::vector<int32_t> sum(static_cast<size_t>(count), 0);
  for (Value partial : op.getPartials()) {
    auto it = memrefs.find(partial);
    if (it == memrefs.end())
      return op.emitError("cim.reduce_partial on an unknown buffer");
    std::vector<uint8_t> bytes;
    if (failed(gather(it->second, bytes)))
      return op.emitError("failed to read a partial sum");
    if (bytes.size() != static_cast<size_t>(count) * 4)
      return op.emitError("partial sum has the wrong size");
    for (int64_t i = 0; i < count; ++i) {
      int32_t value = 0;
      std::memcpy(&value, bytes.data() + i * 4, 4);
      // Wrapping addition, matching the accumulator semantics documented
      // in cimrt: a fixed-width accumulator wraps, it does not saturate.
      sum[static_cast<size_t>(i)] = static_cast<int32_t>(
          static_cast<uint32_t>(sum[static_cast<size_t>(i)]) +
          static_cast<uint32_t>(value));
    }
  }

  cimrt_device *dev = nullptr;
  if (!op.getPartials().empty()) {
    if (auto mvm = op.getPartials().front().getDefiningOp<MvmOp>()) {
      if (auto prog = mvm.getWeights().getDefiningOp<ProgramOp>())
        if (auto ta = prog.getTile().getDefiningOp<TileAllocOp>()) {
          auto it = deviceValues.find(ta.getDevice());
          if (it != deviceValues.end())
            dev = it->second;
        }
    }
  }

  auto alloc =
      makeAllocation(resultType, static_cast<size_t>(count) * 4, dev);
  if (!alloc)
    return op.emitError("failed to allocate the reduction result");
  auto view = viewFromType(resultType, alloc);
  if (failed(view))
    return op.emitError("unsupported layout on the reduction result");

  std::vector<uint8_t> bytes(static_cast<size_t>(count) * 4);
  std::memcpy(bytes.data(), sum.data(), bytes.size());
  if (failed(scatter(*view, bytes)))
    return op.emitError("failed to write the reduction result");

  memrefs[op.getResult()] = *view;
  return success();
}

LogicalResult Interpreter::runCimCopy(CopyOp op) {
  auto srcIt = memrefs.find(op.getSource());
  if (srcIt == memrefs.end())
    return op.emitError("cim.copy on an unknown buffer");

  auto destType = llvm::cast<MemRefType>(op.getDest().getType());
  const unsigned elemBytes = destType.getElementTypeBitWidth() / 8;
  const size_t bytesNeeded =
      static_cast<size_t>(destType.getNumElements()) * elemBytes;

  // A device-space destination needs a device to allocate on. Find one from
  // any device opened so far; cim-partition emits exactly one.
  cimrt_device *dev = devices.empty() ? nullptr : devices.begin()->second;
  auto alloc = makeAllocation(destType, bytesNeeded, dev);
  if (!alloc)
    return op.emitError("failed to allocate the copy destination");

  auto view = viewFromType(destType, alloc);
  if (failed(view))
    return op.emitError("unsupported layout on the copy destination");

  std::vector<uint8_t> bytes;
  if (failed(gather(srcIt->second, bytes)))
    return op.emitError("cim.copy failed to read its source");
  if (bytes.size() != bytesNeeded)
    return op.emitError("cim.copy source and destination differ in size");
  if (failed(scatter(*view, bytes)))
    return op.emitError("cim.copy failed to write its destination");

  memrefs[op.getResult()] = *view;
  return success();
}

LogicalResult Interpreter::runArithConstant(arith::ConstantOp op) {
  // Only scalar integer/index constants are modelled -- what a loop bound
  // or a subview offset can be. A float or vector constant falls through to
  // the Default case in execute() and errors clearly rather than being
  // silently skipped here.
  if (!op.getType().isIndex() && !op.getType().isSignlessInteger())
    return op.emitError(
        "the interpreter only evaluates integer or index constants");
  auto attr = llvm::dyn_cast<IntegerAttr>(op.getValue());
  if (!attr)
    return op.emitError("unsupported arith.constant value");
  indexValues[op.getResult()] = attr.getValue().getSExtValue();
  return success();
}

LogicalResult Interpreter::runScfFor(scf::ForOp op) {
  // Loop-carried values would require merging a value across iterations,
  // which nothing in the pipeline this interpreter executes needs -- every
  // cim op sequence cim-partition emits communicates through side effects
  // (device/tile state, memrefs) rather than through scf.for results.
  if (op.getNumRegionIterArgs() != 0)
    return op.emitError(
        "the interpreter does not support scf.for with loop-carried values "
        "(iter_args); cim-partition never emits any, so a loop that has "
        "them did not come from this pipeline");

  FailureOr<int64_t> lower = evalIndex(op.getLowerBound());
  FailureOr<int64_t> upper = evalIndex(op.getUpperBound());
  FailureOr<int64_t> step = evalIndex(op.getStep());
  if (failed(lower) || failed(upper) || failed(step))
    return op.emitError(
        "scf.for bounds must be values the interpreter can evaluate "
        "(currently only arith.constant); a dynamic trip count cannot be "
        "executed without guessing how many times the loop runs");
  if (*step <= 0)
    return op.emitError("scf.for step must be positive");

  const Value iv = op.getInductionVar();
  Block &body = op.getRegion().front();
  for (int64_t i = *lower; i < *upper; i += *step) {
    indexValues[iv] = i;
    if (failed(executeBlock(body)))
      return failure();
  }
  indexValues.erase(iv);
  return success();
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

LogicalResult Interpreter::executeBlock(Block &block) {
  for (Operation &op : block)
    if (failed(execute(&op)))
      return failure();
  return success();
}

LogicalResult Interpreter::execute(Operation *op) {
  return llvm::TypeSwitch<Operation *, LogicalResult>(op)
      .Case<memref::GetGlobalOp>([&](auto o) { return runGetGlobal(o); })
      .Case<memref::AllocOp, memref::AllocaOp>([&](auto o) {
        return runAlloc(o, llvm::cast<MemRefType>(o.getType()));
      })
      .Case<memref::SubViewOp>([&](auto o) { return runSubView(o); })
      .Case<memref::CopyOp>([&](auto o) { return runMemRefCopy(o); })
      .Case<memref::CastOp>([&](memref::CastOp o) -> LogicalResult {
        // An alias: the cast to an unranked memref for printing keeps the
        // source's sizes and strides.
        auto it = memrefs.find(o.getSource());
        if (it == memrefs.end())
          return o.emitError("cast of an unknown buffer");
        memrefs[o.getResult()] = it->second;
        return success();
      })
      .Case<memref::DeallocOp>([&](auto) { return success(); })
      .Case<func::CallOp>([&](auto o) { return runCall(o); })
      .Case<func::ReturnOp>([&](auto) { return success(); })
      .Case<DeviceOpenOp>([&](auto o) { return runDeviceOpen(o); })
      .Case<TileAllocOp>([&](auto o) { return runTileAlloc(o); })
      .Case<TileFreeOp>([&](auto) { return success(); })
      .Case<ProgramOp>([&](auto o) { return runProgram(o); })
      .Case<MvmOp>([&](auto o) { return runMvm(o); })
      .Case<ReducePartialOp>([&](auto o) { return runReducePartial(o); })
      .Case<CopyOp>([&](auto o) { return runCimCopy(o); })
      .Case<arith::ConstantOp>([&](auto o) { return runArithConstant(o); })
      .Case<scf::ForOp>([&](auto o) { return runScfFor(o); })
      .Case<scf::YieldOp>([&](auto o) -> LogicalResult {
        // A no-arg yield is the natural terminator of every loop body this
        // interpreter runs, since iter_args (and therefore non-trivial
        // yields) are refused in runScfFor.
        if (o.getNumOperands() != 0)
          return o.emitError(
              "scf.yield with operands implies loop-carried values, which "
              "runScfFor already refuses before reaching this terminator");
        return success();
      })
      .Case<BarrierOp>([&](BarrierOp o) -> LogicalResult {
        auto it = deviceValues.find(o.getDevice());
        if (it == deviceValues.end())
          return o.emitError("cim.barrier on an unopened device");
        return success(cimrt_barrier(it->second) == CIMRT_OK);
      })
      .Case<RequantizeOp>([&](RequantizeOp o) -> LogicalResult {
        // Deliberately unimplemented rather than approximated: the rounding
        // mode is not pinned down until cim-legalize-precision defines it,
        // and guessing would produce numbers that look right.
        return o.emitError(
            "cim.requantize is not implemented by the interpreter yet "
            "(cim-legalize-precision has not pinned the rounding mode)");
      })
      .Default([&](Operation *unknown) -> LogicalResult {
        // Never skip silently: an unmodelled op would yield a
        // wrong-but-plausible result, which is exactly what this
        // interpreter exists to catch.
        return unknown->emitError("operation not supported by the cim "
                                  "interpreter: ")
               << unknown->getName();
      });
}

LogicalResult Interpreter::run(ModuleOp mod, StringRef entryName) {
  module = mod;

  auto entry = module.lookupSymbol<func::FuncOp>(entryName);
  if (!entry)
    return module.emitError("no such entry function: ") << entryName;
  if (entry.isExternal())
    return entry.emitError("entry function has no body");
  if (entry.getNumArguments() != 0)
    return entry.emitError(
        "the interpreter runs zero-argument entry functions; supply inputs as "
        "memref.global constants");
  if (!entry.getBody().hasOneBlock())
    return entry.emitError("only single-block functions are supported");

  if (failed(executeBlock(entry.getBody().front())))
    return failure();

  if (options.dumpProfile && !devices.empty()) {
    cimrt_profile profile{};
    if (cimrt_profile_stop(devices.begin()->second, &profile) == CIMRT_OK) {
      out << "cimrt_profile programs=" << profile.programs_issued
          << " mvms=" << profile.mvms_issued
          << " bytes=" << profile.bytes_transferred << "\n";
    }
  }
  return success();
}

} // namespace

LogicalResult mlir::cim::run(ModuleOp module, StringRef entry,
                             const InterpreterOptions &options) {
  Interpreter interp(options, options.out ? *options.out : llvm::outs());
  return interp.run(module, entry);
}
