if(NOT GLSLANG_VALIDATOR OR NOT INPUT OR NOT OUTPUT OR NOT TARGET_ENV)
    message(FATAL_ERROR "Missing required arguments")
endif()

set(TEMP_OUT "${OUTPUT}.tmp")
execute_process(
    COMMAND ${GLSLANG_VALIDATOR} --target-env ${TARGET_ENV} -V ${INPUT} -o ${TEMP_OUT}
    RESULT_VARIABLE COMPILE_RESULT
)
if(NOT COMPILE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to compile ${INPUT}")
endif()

if(OPTIMIZE AND SPIRV_OPT)
    set(OPT_OUT "${OUTPUT}.opt")
    execute_process(
        COMMAND ${SPIRV_OPT} -Os --strip-debug ${TEMP_OUT} -o ${OPT_OUT}
        RESULT_VARIABLE OPT_RESULT
    )
    if(NOT OPT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to optimize ${INPUT}")
    endif()
    file(REMOVE ${TEMP_OUT})
    file(RENAME ${OPT_OUT} ${OUTPUT})
else()
    file(RENAME ${TEMP_OUT} ${OUTPUT})
endif()