PROG:=multi_gpu_support

HALIDE_ROOT?=../..
HALIDE_BUILD?=${HALIDE_ROOT}

HALIDE_TOOLS_DIR=${HALIDE_ROOT}/tools
HALIDE_APPS_DIR=${HALIDE_ROOT}/bin/apps
HALIDE_LIB_CMAKE:=${HALIDE_BUILD}/lib
HALIDE_LIB_MAKE:=${HALIDE_BUILD}/bin

HALIDE_LIB:=libHalide.so
BUILD_BY_CMAKE:=$(shell ls ${HALIDE_LIB_CMAKE} | grep ${HALIDE_LIB})
BUILD_BY_MAKE:=$(shell ls ${HALIDE_LIB_MAKE} | grep ${HALIDE_LIB})

ifeq (${BUILD_BY_CMAKE}, ${HALIDE_LIB})
	HALIDE_LIB_DIR=${HALIDE_LIB_CMAKE}
else ifeq (${BUILD_BY_MAKE}, ${HALIDE_LIB})
	HALIDE_LIB_DIR=${HALIDE_LIB_MAKE}
endif

CUDA_INCLUDE=/usr/local/cuda/include
CUDA_LIBS=-L/usr/lib/nvidia-current -lcuda

GENFLAGS= -fno-rtti -O3 -g -std=c++11 -fno-omit-frame-pointer -fno-rtti -Wall -Wno-unused-function -Wcast-qual -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-psabi
CXXFLAGS= -std=c++11 -g -fno-omit-frame-pointer -fno-rtti -Wall -Werror -Wcast-qual -Wno-unused-variable -Wignored-qualifiers -Wno-comment -Wsign-compare -Wno-unknown-warning-option -Wno-psabi -march=native -Wno-unknown-pragmas

all: ${PROG}_test

${PROG}_gen: ${PROG}_generator.cpp
	g++ ${GENFLAGS} -I${HALIDE_ROOT}/include -I${HALIDE_TOOLS_DIR} -L${HALIDE_LIB_DIR} -DUSE_HALIDE -I${CUDA_INCLUDE} $< ${HALIDE_TOOLS_DIR}/GenGen.cpp -o ${PROG}_gen -ldl -lpthread -lz -lHalide
  
${PROG}_gen.exec: ${PROG}_gen
	LD_LIBRARY_PATH=${HALIDE_LIB_DIR} ./$< -o . -g ${PROG} -e h,static_library,html target=host-cuda-no_asserts-user_context

${PROG}.a: ${PROG}_gen.exec

${PROG}.h: ${PROG}_gen.exec

${PROG}_test: ${PROG}_test.cpp ${PROG}.h ${PROG}.a
	g++ ${CXXFLAGS} -I${CUDA_INCLUDE} -I${HALIDE_TOOLS_DIR} -DTEST_CUDA $< ${PROG}.a -I${HALIDE_ROOT} -I${HALIDE_ROOT}/apps/support -I${HALIDE_ROOT}/src/runtime -I${HALIDE_TOOLS_DIR} ${CUDA_LIBS} -ldl -lpthread -lz -o $@

test: ${PROG}_test
	./${PROG}_test
		 
clean:
	rm -rf ${PROG}_gen ${PROG}_test ${PROG}_*test_csim ${PROG}_run ${PROG}*.h ${PROG}*.a ${PROG}*.html *.o *.hls *.exec *.dSYM *.ppm *.pgm *.dat

