INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)
CMAKE_ARGS=${TRANSFORMERS_CMAKE_ARGS}
EXTRA_OBJS =

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifndef UNAME_P
UNAME_P := $(shell uname -p)
endif

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

CCV := $(shell $(CC) --version | head -n 1)
CXXV := $(shell $(CXX) --version | head -n 1)

# Mac OS + Arm can report x86_64
# ref: https://github.com/ggerganov/whisper.cpp/issues/66#issuecomment-1282546789
ifeq ($(UNAME_S),Darwin)
	ifneq ($(UNAME_P),arm)
		SYSCTL_M := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
		ifeq ($(SYSCTL_M),1)
			# UNAME_P := arm
			# UNAME_M := arm64
			warn := $(warning Your arch is announced as x86_64, but it seems to actually be ARM64. Not fixing that can lead to bad performance. For more info see: https://github.com/ggerganov/whisper.cpp/issues/66\#issuecomment-1282546789)
		endif
	endif
endif

#
# Compile flags
#

# keep standard at C11 and C++11
CFLAGS   = -I. -I./ggml.cpp/include -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -I -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I. -I./ggml.cpp/include -I./ggml.cpp/include/ggml/ -I./ggml.cpp/examples/ -O3 -DNDEBUG -std=c++17 -fPIC
LDFLAGS  =

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar

# OS specific
# TODO: support Windows
ifeq ($(UNAME_S),Linux)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Darwin)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),FreeBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),NetBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),OpenBSD)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif
ifeq ($(UNAME_S),Haiku)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	# Use all CPU extensions that are available:
	CFLAGS += -march=native -mtune=native
	CXXFLAGS += -march=native -mtune=native
endif
ifneq ($(filter ppc64%,$(UNAME_M)),)
	POWER9_M := $(shell grep "POWER9" /proc/cpuinfo)
	ifneq (,$(findstring POWER9,$(POWER9_M)))
		CFLAGS += -mcpu=power9
		CXXFLAGS += -mcpu=power9
	endif
	# Require c++23's std::byteswap for big-endian support.
	ifeq ($(UNAME_M),ppc64)
		CXXFLAGS += -std=c++23 -DGGML_BIG_ENDIAN
	endif
endif
ifndef LLAMA_NO_ACCELERATE
	# Mac M1 - include Accelerate framework.
	# `-framework Accelerate` works on Mac Intel as well, with negliable performance boost (as of the predict time).
	ifeq ($(UNAME_S),Darwin)
		CFLAGS  += -DGGML_USE_ACCELERATE
		LDFLAGS += -framework Accelerate
	endif
endif
ifdef LLAMA_OPENBLAS
	CFLAGS  += -DGGML_USE_OPENBLAS -I/usr/local/include/openblas
	LDFLAGS += -lopenblas
endif
ifdef LLAMA_GPROF
	CFLAGS   += -pg
	CXXFLAGS += -pg
endif
ifneq ($(filter aarch64%,$(UNAME_M)),)
	CFLAGS += -mcpu=native
	CXXFLAGS += -mcpu=native
endif
ifneq ($(filter armv6%,$(UNAME_M)),)
	# Raspberry Pi 1, 2, 3
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access
endif
ifneq ($(filter armv7%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfpu=neon-fp-armv8 -mfp16-format=ieee -mno-unaligned-access -funsafe-math-optimizations
endif
ifneq ($(filter armv8%,$(UNAME_M)),)
	# Raspberry Pi 4
	CFLAGS += -mfp16-format=ieee -mno-unaligned-access
endif

ifeq ($(BUILD_TYPE),cublas)
	CMAKE_ARGS += -DGGML_CUBLAS=ON
	EXTRA_OBJS += ggml-cuda.o
endif

#
# Print build information
#

$(info I libtransformers build info: )
$(info I UNAME_S:  $(UNAME_S))
$(info I UNAME_P:  $(UNAME_P))
$(info I UNAME_M:  $(UNAME_M))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CMAKE_ARGS:  $(CMAKE_ARGS))
$(info I CC:       $(CCV))
$(info I CXX:      $(CXXV))
$(info )


ggml.o:
	mkdir build
	cd build && cmake ../ggml.cpp $(CMAKE_ARGS) && make VERBOSE=1 ggml && cp -rf src/CMakeFiles/ggml.dir/ggml.c.o ../ggml.o
	@if [ "$(BUILD_TYPE)" = "cublas" ]; then \
		cp -rf build/src/CMakeFiles/ggml.dir/ggml-cuda.cu.o ggml-cuda.o ;\
	fi

generic-ggml.o:
	$(CC) $(CFLAGS) -c ggml.cpp/src/ggml.c -o ggml.o

common.o:
	$(CXX) $(CXXFLAGS) -c ggml.cpp/examples/common.cpp -o common.o

common-ggml.o:
	$(CXX) $(CXXFLAGS) -c ggml.cpp/examples/common-ggml.cpp -o common-ggml.o

clean:
	rm -f *.o
	rm -f *.a
	rm -rf build
	rm -rf example
	rm -f main

bloom: bloomz.cpp ggml.o common.o
	$(CXX) $(CXXFLAGS) bloomz.cpp ggml.o common.o -o main $(LDFLAGS)
	./main -h

gpt2.o: gpt2.cpp ggml.o common.o
	$(CXX) $(CXXFLAGS) gpt2.cpp ggml.o common.o -o main -c $(LDFLAGS)

falcon.o: falcon.cpp
	$(CXX) $(CXXFLAGS) falcon.cpp -o falcon.o -c $(LDFLAGS)

dolly.o: dolly.cpp
	$(CXX) $(CXXFLAGS) dolly.cpp -o dolly.o -c $(LDFLAGS)

replit.o: replit.cpp
	$(CXX) $(CXXFLAGS) replit.cpp -o replit.o -c $(LDFLAGS)

mpt.o: mpt.cpp
	$(CXX) $(CXXFLAGS) mpt.cpp -o mpt.o -c $(LDFLAGS)

gptj.o: gptj.cpp
	$(CXX) $(CXXFLAGS) gptj.cpp -o gptj.o -c $(LDFLAGS)

gptneox.o: gptneox.cpp
	$(CXX) $(CXXFLAGS) gptneox.cpp -o gptneox.o -c $(LDFLAGS)

starcoder.o: starcoder.cpp
	$(CXX) $(CXXFLAGS) starcoder.cpp -o starcoder.o -c $(LDFLAGS)

prepare:
# As we are going to link back to the examples, we need to avoid multiple definitions of main.
# This is hack, but it's easier to maintain than duplicating code from the main repository.
	@find ./ggml.cpp/examples/dolly-v2 -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_dolly/g' {} +
	@find ./ggml.cpp/examples/gpt-2 -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_gpt2/g' {} +
	@find ./ggml.cpp/examples/gpt-j -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_gptj/g' {} +
	@find ./ggml.cpp/examples/gpt-neox -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_gptneox/g' {} +
	@find ./ggml.cpp/examples/mpt -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_mpt/g' {} +
	@find ./ggml.cpp/examples/replit -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_replit/g' {} +
	@find ./ggml.cpp/examples/starcoder -type f -name "*.cpp" -exec sed -i'' -e 's/int main/int main_starcoder/g' {} +

libtransformers.a: prepare starcoder.o falcon.o gptj.o mpt.o gpt2.o replit.o gptneox.o ggml.o dolly.o common-ggml.o common.o
	ar src libtransformers.a replit.o gptj.o mpt.o gptneox.o starcoder.o gpt2.o dolly.o  falcon.o  ggml.o common-ggml.o common.o ${EXTRA_OBJS}

example:
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go build -o example -x ./examples

test: libtransformers.a
	@C_INCLUDE_PATH=${INCLUDE_PATH} LIBRARY_PATH=${LIBRARY_PATH} go test -v ./...