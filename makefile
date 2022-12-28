Exec = main
OPTS = -O3 -std=c++11 -g

NVCC=nvcc

objects := $(patsubst %.cpp,%.o,$(wildcard *.cpp))
cu_objects := $(patsubst %.cu,%.o,$(wildcard *.cu))


.PHONY: all
all: $(Exec)

.PHONY: check
check:
	echo Objects are $(objects)
	echo Cu_Objects are $(cu_objects)

$(objects): %.o: %.cpp *.h
	$(NVCC) -c $(OPTS) $< -o $@

$(cu_objects): %.o: %.cu *.cudah
	$(NVCC) -c $(OPTS) $< -o $@
	

$(Exec): $(objects) $(cu_objects)
	$(NVCC) $(OPTS) $(objects) $(cu_objects) -o $(Exec)
	echo $(NVCC) $(OPTS) $(objects) $(cu_objects) -o $(Exec)

.PHONY: clean
clean:
	-rm *.o $(Exec)