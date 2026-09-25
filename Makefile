NVCC ?= nvcc
ARCH ?= sm_75
MMA ?= auto
CASE ?= smoke
CONFIG ?= workloads.csv
WITH_CUDNN ?= 0
CUDNN_ROOT ?= /usr

BUILD := build
BIN := $(BUILD)/conv_lab
CUDNN_BIN := $(BUILD)/cudnn_lab
CASE_HEADER := $(BUILD)/case_config.cuh
HEADERS := $(wildcard include/*.cuh)
# Compare the compute capability numerically.  A prefix match on sm_8%/sm_9%
# silently missed sm_10x and sm_12x and picked the m8n8k16 path on Blackwell,
# which is both ~3x slower and ranks the dispatch candidates differently.
ARCH_NUM := $(shell echo '$(ARCH)' | sed 's/[^0-9]//g')
ifeq ($(MMA),auto)
MMA := $(shell test '$(ARCH_NUM)' -ge 80 2>/dev/null && echo sm80 || echo sm75)
endif

ifeq ($(MMA),sm80)
MMAFLAGS := -DINT8_LAB_MMA_SM80=1
else ifeq ($(MMA),sm75)
MMAFLAGS := -DINT8_LAB_MMA_SM80=0
else
$(error MMA must be auto, sm75, or sm80)
endif

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Iinclude -I$(BUILD) $(MMAFLAGS)
ifeq ($(WITH_CUDNN),1)
NVCCFLAGS += -DINT8_LAB_WITH_CUDNN=1 -I$(CUDNN_ROOT)/include
LDFLAGS := -L$(CUDNN_ROOT)/lib -Xlinker=-rpath -Xlinker=$(CUDNN_ROOT)/lib -lcudnn
else
NVCCFLAGS += -DINT8_LAB_WITH_CUDNN=0
endif

.PHONY: all run clean force
all: $(BIN) $(if $(filter 1,$(WITH_CUDNN)),$(CUDNN_BIN))

$(CASE_HEADER): force scripts/gen_case.py $(CONFIG)
	@mkdir -p $(BUILD)
	python3 scripts/gen_case.py $(CONFIG) $(CASE) $@

$(BIN): src/conv_lab.cu $(HEADERS) $(CASE_HEADER)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -include $(CASE_HEADER) $< -o $@ $(LDFLAGS)

$(CUDNN_BIN): src/cudnn_lab.cu $(HEADERS) $(CASE_HEADER)
	@mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) -include $(CASE_HEADER) $< -o $@ $(LDFLAGS)

run: $(BIN)
	./$(BIN) --strategy both

clean:
	rm -rf $(BUILD)

force:
