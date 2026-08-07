# Opt-in only (CIM_ENABLE_REAL_TARGET_E2E, default OFF): drives
# cim-lower-to-target's output through mlir-opt's --convert-to-llvm
# pipeline, mlir-translate, and clang, then links the result against the
# real runtime/libcimrt.a and RUNS it -- the same manual round trip
# documented in lib/Transforms/CIMLowerToTarget.cpp's file header, now
# reproducible with one build instead of by hand.
#
# Two variants of the same MLIR source (see check.mlir.in), differing only
# in the constant a cf.assert checks a real mvm result against:
#   correct -- the true value; the binary must exit 0.
#   wrong   -- a deliberately false value; cf.assert must fire, so the
#              binary must genuinely crash (SIGABRT), not exit cleanly.
# Asserting both directions is the point: a verification step that always
# "passes" because nothing would ever make it fail is not verification.

find_program(CIM_REAL_TARGET_CLANG NAMES clang++-18 clang++ clang-18 clang)
if(NOT CIM_REAL_TARGET_CLANG)
  message(FATAL_ERROR
    "CIM_ENABLE_REAL_TARGET_E2E is ON but no clang was found on PATH -- "
    "this target needs clang specifically to compile the .ll mlir-translate "
    "emits (not just any C/C++ compiler CMake configured). Install "
    "clang-18 or pass -DCIM_REAL_TARGET_CLANG=<path>.")
endif()

if(NOT EXISTS "${LLVM_TOOLS_BINARY_DIR}/mlir-opt")
  message(FATAL_ERROR
    "CIM_ENABLE_REAL_TARGET_E2E is ON but mlir-opt was not found in "
    "${LLVM_TOOLS_BINARY_DIR}. This target needs the same LLVM/MLIR install "
    "cim-opt itself was configured against.")
endif()
if(NOT EXISTS "${LLVM_TOOLS_BINARY_DIR}/mlir-translate")
  message(FATAL_ERROR
    "CIM_ENABLE_REAL_TARGET_E2E is ON but mlir-translate was not found in "
    "${LLVM_TOOLS_BINARY_DIR}.")
endif()

set(CIM_REAL_TARGET_YAML "${PROJECT_SOURCE_DIR}/test/targets/tiny-4x4.yaml")

# Runs the whole cim-opt -> mlir-opt -> mlir-translate -> clang chain for
# one variant (NAME is "correct" or "wrong"), producing an object file at
# ${CIM_REAL_TARGET_OBJ_<NAME>} that add_real_target_binary links below.
# One function, not copy-pasted per variant, so the two variants cannot
# silently drift apart in how they are built -- only in EXPECTED_VALUE,
# which is the whole point of having two.
function(add_real_target_object NAME EXPECTED_VALUE)
  set(configured_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.mlir")
  set(lowered_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.lowered.mlir")
  set(llvm_dialect_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.llvmdialect.mlir")
  set(ll_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.ll")
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")

  configure_file(check.mlir.in ${configured_src} @ONLY)

  add_custom_command(
    OUTPUT ${lowered_src}
    COMMAND $<TARGET_FILE:cim-opt> ${configured_src}
            --cim-lower-to-target=target-yaml=${CIM_REAL_TARGET_YAML}
            -o ${lowered_src}
    DEPENDS cim-opt ${configured_src}
    COMMENT "real-target-e2e (${NAME}): cim-lower-to-target")

  add_custom_command(
    OUTPUT ${llvm_dialect_src}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${lowered_src}
            --finalize-memref-to-llvm --convert-func-to-llvm
            --convert-arith-to-llvm --convert-cf-to-llvm
            --reconcile-unrealized-casts
            -o ${llvm_dialect_src}
    DEPENDS ${lowered_src}
    COMMENT "real-target-e2e (${NAME}): --convert-to-llvm")

  add_custom_command(
    OUTPUT ${ll_src}
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-translate --mlir-to-llvmir
            ${llvm_dialect_src} -o ${ll_src}
    DEPENDS ${llvm_dialect_src}
    COMMENT "real-target-e2e (${NAME}): mlir-translate")

  add_custom_command(
    OUTPUT ${obj}
    COMMAND ${CIM_REAL_TARGET_CLANG} -c ${ll_src} -o ${obj}
    DEPENDS ${ll_src}
    COMMENT "real-target-e2e (${NAME}): clang -c")

  set(CIM_REAL_TARGET_OBJ_${NAME} ${obj} PARENT_SCOPE)
endfunction()

add_real_target_object(correct 7)
add_real_target_object(wrong 999)

foreach(variant correct wrong)
  add_executable(real-target-e2e-${variant}
    driver.c
    ${CIM_REAL_TARGET_OBJ_${variant}})
  # driver.c is real C; the ${variant}.o came from mlir-translate's LLVM IR
  # and needs no compiling, only linking -- CMake treats a source whose
  # extension it does not recognize as an object to link as-is.
  set_source_files_properties(${CIM_REAL_TARGET_OBJ_${variant}}
    PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
  target_link_libraries(real-target-e2e-${variant} PRIVATE cimrt cimTarget)
endforeach()

add_test(NAME real-target-e2e-correct-value-passes
         COMMAND real-target-e2e-correct)

# The whole point of this variant: cf.assert firing inside the compiled
# binary is a real SIGABRT, not just a nonzero exit -- and CTest's WILL_FAIL
# property explicitly does NOT invert a signal-terminated ("system-level")
# failure (`cmake --help-property WILL_FAIL`), so it cannot express "this
# test passes when the binary crashes". `sh -c '! ...'` inverts the shell's
# own exit status instead: a crash (nonzero/signal) makes `!`'s result 0,
# a genuine PASS from ctest's point of view, and if the binary ever exits 0
# (meaning the wrong-value check silently stopped checking anything) `!`
# makes the wrapper fail, a genuine, ordinary CTest FAILED.
add_test(NAME real-target-e2e-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-wrong>")
