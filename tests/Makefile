NVCC=nvcc

INCDIR=../include

NVCCFLAGS=-std=c++17 -I$(INCDIR) -Xcompiler="-fopenmp" --ptxas-options=-v
NVCCFLAGS+=-gencode arch=compute_86,code=sm_86
NVCCFLAGS+=-gencode arch=compute_80,code=sm_80
#NVCCFLAGS+=-gencode arch=compute_75,code=sm_75
#NVCCFLAGS+=-gencode arch=compute_70,code=sm_70

TARGET=mma.test elementwise.test

all: $(TARGET)

%.test:%.cu
	$(NVCC) $< $(OBJS) $(NVCCFLAGS) -o $@

clean:
	rm -f *.test
