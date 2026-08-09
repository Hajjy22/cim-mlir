# Opt-in only (CIM_ENABLE_REAL_TARGET_E2E, default OFF): drives
# cim-lower-to-target's output through mlir-opt's --convert-to-llvm
# pipeline, mlir-translate, and clang, then links the result against the
# real runtime/libcimrt.a and RUNS it -- the same manual round trip
# documented in lib/Transforms/CIMLowerToTarget.cpp's file header, now
# reproducible with one build instead of by hand.
#
# Several variant PAIRS, each built from its own MLIR template (check.mlir.in
# for cim.mvm, check-requantize.mlir.in for cim.requantize, ...  -- see
# each template's own header), differing only in the constant a cf.assert
# checks a real result against:
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
# Same 2-tiles-of-4x4 geometry as tiny-4x4.yaml, differing only in
# persistence (persistent: false, persistence: volatile) -- see that
# file's own header. Used by exactly one variant pair below, to prove the
# whole chain down to a real compiled binary does not silently assume a
# nonvolatile target (docs/roadmap.md's M4 "prove retargetability" entry).
set(CIM_DIGITAL_CIM_TARGET_YAML
    "${PROJECT_SOURCE_DIR}/test/targets/tiny-digital-cim.yaml")

# Runs the whole cim-opt -> mlir-opt -> mlir-translate -> clang chain for
# one variant (NAME, e.g. "correct" or "requantize-wrong"), configured from
# TEMPLATE, producing an object file at ${CIM_REAL_TARGET_OBJ_<NAME>} that
# the executable loop below links. One function, not copy-pasted per
# variant, so variants sharing a template cannot silently drift apart in
# how they are built -- only in EXPECTED_VALUE, which is the whole point
# of having two per template.
function(add_real_target_object NAME TEMPLATE EXPECTED_VALUE)
  # A 4th, optional argument overrides which target file this one variant
  # is built against; every existing call site omits it and gets the
  # default. set() with no PARENT_SCOPE only shadows CIM_REAL_TARGET_YAML
  # for the rest of THIS function call, so the override never leaks to
  # variants added after it.
  if(ARGC GREATER 3)
    set(CIM_REAL_TARGET_YAML ${ARGV3})
  endif()

  set(configured_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.mlir")
  set(lowered_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.lowered.mlir")
  set(llvm_dialect_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.llvmdialect.mlir")
  set(ll_src "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.ll")
  set(obj "${CMAKE_CURRENT_BINARY_DIR}/${NAME}.o")

  configure_file(${TEMPLATE} ${configured_src} @ONLY)

  add_custom_command(
    OUTPUT ${lowered_src}
    # cim-detect/cim-partition/cim-placement all run ahead of
    # cim-lower-to-target for every variant, not just check-loop.mlir.in/
    # check-multi-k-tile.mlir.in below: they are true no-ops on a template
    # with no linalg.matmul_transpose_b in it (detect finds no candidate,
    # partition has nothing marked to rewrite, placement has no
    # cim.program to reuse or hoist), so this is free for
    # check.mlir.in/check-requantize.mlir.in/check-reduce-partial.mlir.in
    # and is what lets check-loop.mlir.in/check-multi-k-tile.mlir.in start
    # from a real linalg op -- the actual compiler front door -- like a
    # real caller would, rather than hand-writing cim ops cim-partition
    # (and, for a multi-row matmul, cim-placement's own loop hoisting --
    # see check-loop.mlir.in's own header) would have produced.
    COMMAND $<TARGET_FILE:cim-opt> ${configured_src}
            --cim-detect
            --cim-partition=target-yaml=${CIM_REAL_TARGET_YAML}
            --cim-placement=target-yaml=${CIM_REAL_TARGET_YAML}
            --cim-lower-to-target=target-yaml=${CIM_REAL_TARGET_YAML}
            -o ${lowered_src}
    DEPENDS cim-opt ${configured_src}
    COMMENT "real-target-e2e (${NAME}): cim-detect/cim-partition/cim-placement/cim-lower-to-target")

  add_custom_command(
    OUTPUT ${llvm_dialect_src}
    # --expand-strided-metadata ahead of --finalize-memref-to-llvm: without
    # it, a rank-2 host memref.alloc that is both memref.copy'd into and
    # memref.subview'd (the shape check-multi-k-tile.mlir.in's staged
    # activation buffer needs -- see its own header) leaves a dangling
    # unrealized_conversion_cast --reconcile-unrealized-casts cannot clear,
    # because FinalizeMemRefToLLVM's own memref.copy pattern does not
    # otherwise resolve the subview's strided layout back to the alloc's
    # plain one. The other, simpler templates never exercised that
    # combination, which is why this was not needed until now.
    #
    # --convert-scf-to-cf ahead of everything else, and --lower-affine
    # ahead of THAT: check-loop.mlir.in is the first template whose
    # cim-lower-to-target output still has a real scf.for in it (a loop
    # body cim-lower-to-target lowered in place -- see its file header
    # point 1), and --expand-strided-metadata turns that loop's
    # per-iteration memref.subview (a dynamic, per-row offset) into an
    # affine.apply computing the byte offset, which nothing else here
    # converts to something --convert-to-llvm's interfaces understand
    # -- --lower-affine is what does. Both must run before
    # --convert-cf-to-llvm specifically: converting cf.assert into
    # multi-block LLVM-dialect control flow before scf.for's own single
    # block has been converted to blocks itself leaves a loop region with
    # more than the one block scf.for's own verifier allows.
    COMMAND ${LLVM_TOOLS_BINARY_DIR}/mlir-opt ${lowered_src}
            --expand-strided-metadata --lower-affine --convert-scf-to-cf
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

# cim.mvm: identity weight, out[0] == act[0] == 7 by construction.
add_real_target_object(correct check.mlir.in 7)
add_real_target_object(wrong check.mlir.in 999)
# cim.requantize: a raw dot product (40000) far outside an 8-bit signed
# clamp, so reaching the expected value needs a real, working
# cimrt_requantize call, not just a value passed through unchanged -- see
# check-requantize.mlir.in's own header.
add_real_target_object(requantize-correct check-requantize.mlir.in 127)
add_real_target_object(requantize-wrong check-requantize.mlir.in 999)
# cim.reduce_partial: two independent partial sums (10 and 15) that must
# actually be added to reach 25 -- see check-reduce-partial.mlir.in's own
# header for why neither partial alone can pass this check.
add_real_target_object(reduce-partial-correct check-reduce-partial.mlir.in 25)
add_real_target_object(reduce-partial-wrong check-reduce-partial.mlir.in 999)
# A genuine multi-K-tile linalg.matmul_transpose_b through the real
# cim-detect/cim-partition/cim-lower-to-target chain -- see
# check-multi-k-tile.mlir.in's own header for why 36 is the value that
# proves both K-tiles' contributions were actually combined.
add_real_target_object(multi-k-tile-correct check-multi-k-tile.mlir.in 36)
add_real_target_object(multi-k-tile-wrong check-multi-k-tile.mlir.in 999)
# A genuine multi-ROW linalg.matmul_transpose_b, through cim-placement's
# own loop hoisting this time (not just cim-detect/cim-partition) -- see
# check-loop.mlir.in's own header for why 951 is the checksum that proves
# all three loop iterations independently computed and kept their own
# correct per-row result.
add_real_target_object(loop-correct check-loop.mlir.in 951)
add_real_target_object(loop-wrong check-loop.mlir.in 999)
# A genuine multi-ROW linalg.matmul_transpose_b, same checksum scheme as
# loop-correct/-wrong above, but through cim-partition's OWN generated
# scf.for this time -- see check-batch.mlir.in's own header for why 951 is
# the checksum that proves the loop this pass wrote, not one a caller
# hand-wrote for it, keeps each row's result independent and correct.
add_real_target_object(batch-correct check-batch.mlir.in 951)
add_real_target_object(batch-wrong check-batch.mlir.in 999)
# A genuine linalg.matmul (NOT matmul_transpose_b) -- see
# check-plain-matmul.mlir.in's own header for why 4321 is the checksum
# that proves cim-partition's own generated weight transpose is correct,
# not merely structurally plausible.
add_real_target_object(plain-matmul-correct check-plain-matmul.mlir.in 4321)
add_real_target_object(plain-matmul-wrong check-plain-matmul.mlir.in 999)
# A doubly-nested, hand-written PLAIN scf.for (no iter_args at either
# level) -- cim-lower-to-target's newest lowering capability, one level
# deeper than check-loop.mlir.in's single-level, cim-placement-produced
# loop (cim-placement itself never nests two levels, so this is hand-
# written the way check.mlir.in is, not rewritten from a real matmul) --
# see check-nested-loop.mlir.in's own header for why 654321 is the
# checksum that proves all six (outer, inner) iterations, across both
# loop levels, independently computed and kept their own correct result.
add_real_target_object(nested-loop-correct check-nested-loop.mlir.in 654321)
add_real_target_object(nested-loop-wrong check-nested-loop.mlir.in 999)
# Same template and expected value as correct/wrong above (identity
# weight, out[0] == act[0] == 7 by construction) -- the only thing this
# pair changes is the target file, to a volatile one. If cim-partition,
# cim-lower-to-target, or cimrt_program ever silently assumed
# `persistent: true`, this is what would catch it: not a FileCheck
# assertion on an attribute, a real binary that must still compute the
# right answer.
add_real_target_object(digital-cim-volatile-correct check.mlir.in 7
                       ${CIM_DIGITAL_CIM_TARGET_YAML})
add_real_target_object(digital-cim-volatile-wrong check.mlir.in 999
                       ${CIM_DIGITAL_CIM_TARGET_YAML})

foreach(variant correct wrong requantize-correct requantize-wrong
                reduce-partial-correct reduce-partial-wrong
                multi-k-tile-correct multi-k-tile-wrong
                loop-correct loop-wrong
                batch-correct batch-wrong
                plain-matmul-correct plain-matmul-wrong
                nested-loop-correct nested-loop-wrong
                digital-cim-volatile-correct digital-cim-volatile-wrong)
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
add_test(NAME real-target-e2e-requantize-correct-value-passes
         COMMAND real-target-e2e-requantize-correct)
add_test(NAME real-target-e2e-reduce-partial-correct-value-passes
         COMMAND real-target-e2e-reduce-partial-correct)
# Needs WORKING_DIRECTORY, unlike every other variant here: those all
# hand-write cim.device_open with @CIM_REAL_TARGET_YAML@ (an absolute
# path) substituted directly into the target string. check-multi-k-tile
# .mlir.in has no cim.device_open of its own -- cim-partition inserts one
# itself, using the target file's own `name:` field ("tiny-4x4"), which
# cimrt_open then resolves as a RELATIVE path ("targets/tiny-4x4.yaml")
# against the process's current directory. tiny-4x4.yaml is a test-only
# fixture that lives under test/targets/, not the top-level targets/ real
# target files resolve against, so the working directory has to be
# test/, specifically -- the repo root is not enough (there is no
# top-level targets/tiny-4x4.yaml to find there).
add_test(NAME real-target-e2e-multi-k-tile-correct-value-passes
         COMMAND real-target-e2e-multi-k-tile-correct
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
# Same WORKING_DIRECTORY reason as multi-k-tile above: check-loop.mlir.in
# also has no cim.device_open of its own -- cim-partition inserts one.
add_test(NAME real-target-e2e-loop-correct-value-passes
         COMMAND real-target-e2e-loop-correct
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
# Same WORKING_DIRECTORY reason as loop above: check-batch.mlir.in also has
# no cim.device_open of its own -- cim-partition inserts one.
add_test(NAME real-target-e2e-batch-correct-value-passes
         COMMAND real-target-e2e-batch-correct
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
# Same WORKING_DIRECTORY reason as batch above: check-plain-matmul.mlir.in
# also has no cim.device_open of its own -- cim-partition inserts one.
add_test(NAME real-target-e2e-plain-matmul-correct-value-passes
         COMMAND real-target-e2e-plain-matmul-correct
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
# check-nested-loop.mlir.in hand-writes cim.device_open with the absolute
# path substituted in directly (see check.mlir.in's own correct/wrong
# above), so this needs no WORKING_DIRECTORY either.
add_test(NAME real-target-e2e-nested-loop-correct-value-passes
         COMMAND real-target-e2e-nested-loop-correct)
# check.mlir.in hand-writes cim.device_open with the absolute path
# substituted in directly (see correct/wrong above), so this needs no
# WORKING_DIRECTORY -- same reasoning as correct-value-passes itself.
add_test(NAME real-target-e2e-digital-cim-volatile-correct-value-passes
         COMMAND real-target-e2e-digital-cim-volatile-correct)

# The whole point of these two variants: cf.assert firing inside the
# compiled binary is a real SIGABRT, not just a nonzero exit -- and CTest's
# WILL_FAIL property explicitly does NOT invert a signal-terminated
# ("system-level") failure (`cmake --help-property WILL_FAIL`), so it
# cannot express "this test passes when the binary crashes". `sh -c
# '! ...'` inverts the shell's own exit status instead: a crash
# (nonzero/signal) makes `!`'s result 0, a genuine PASS from ctest's point
# of view, and if the binary ever exits 0 (meaning the wrong-value check
# silently stopped checking anything) `!` makes the wrapper fail, a
# genuine, ordinary CTest FAILED.
add_test(NAME real-target-e2e-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-wrong>")
add_test(NAME real-target-e2e-requantize-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-requantize-wrong>")
add_test(NAME real-target-e2e-reduce-partial-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-reduce-partial-wrong>")
add_test(NAME real-target-e2e-multi-k-tile-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-multi-k-tile-wrong>"
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
add_test(NAME real-target-e2e-loop-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-loop-wrong>"
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
add_test(NAME real-target-e2e-batch-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-batch-wrong>"
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
add_test(NAME real-target-e2e-plain-matmul-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-plain-matmul-wrong>"
         WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test)
add_test(NAME real-target-e2e-nested-loop-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-nested-loop-wrong>")
add_test(NAME real-target-e2e-digital-cim-volatile-wrong-value-traps
         COMMAND sh -c "! $<TARGET_FILE:real-target-e2e-digital-cim-volatile-wrong>")
