# Non-volatile memory simulator

#target := tsvtest
target := destiny
model_target := scope_model

# define tool chain
CXX := g++
RM := rm -f

# define build options
# compile options
CXXFLAGS := -std=c++17 -Wall
# link options
LDFLAGS :=
# link librarires
LDLIBS :=

OUTDIR := obj

# construct list of .cpp and their corresponding .o and .d files
SRC := main.cpp $(wildcard component/*.cpp) $(wildcard model/*.cpp)
INC := -I. -Icomponent -Imodel
DBG :=
OBJ := $(OUTDIR)/main.o $(patsubst component/%.cpp,$(OUTDIR)/%.o,$(wildcard component/*.cpp))
MODEL_OBJ := $(patsubst model/%.cpp,$(OUTDIR)/model_%.o,$(wildcard model/*.cpp))
DEP := Makefile.dep

# file disambiguity is achieved via the .PHONY directive
.PHONY : all clean dbg scope scope-v4 scope-v5 scope-v6 scope-requested test-scope

all: CXXFLAGS += -O3 -mtune=native
all: dir $(target) $(model_target)

dbg: DBG += -ggdb -g #-DNVSIM3DDEBUG=1
dbg: dir $(target)

dir:
	mkdir -p $(OUTDIR)

$(target): $(OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(model_target): $(MODEL_OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) $(target) $(model_target) $(DEP) $(OBJ) $(MODEL_OBJ)

scope: all
	python3 scope.py config/scope_v3.json --json-output results/scope_v3.json

scope-v4: all
	python3 scope.py config/scope_v4.json --explore --json-output results/scope_v4_attention.json

scope-v5: all
	python3 scope.py config/scope_v5.json --explore --json-output results/scope_v5_attention.json
	python3 scope.py config/scope_v5.json --workload ffn --explore --json-output results/scope_v5_ffn.json

scope-v6: all
	python3 scope.py config/scope_v6.json --explore --json-output results/scope_v6_attention.json
	python3 scope.py config/scope_v6.json --workload ffn --explore --json-output results/scope_v6_ffn.json

scope-requested: all
	python3 scope.py config/scope_v2_requested.json --json-output results/scope_v2_requested.json

test-scope: $(model_target)
	python3 -m unittest discover -s tests -v

$(OUTDIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

$(OUTDIR)/%.o: component/%.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

$(OUTDIR)/model_%.o: model/%.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

depend $(DEP):
	@echo Makefile - creating dependencies for: $(SRC)
	@$(RM) $(DEP)
	@$(CXX) -E -MM $(INC) $(SRC) >> $(DEP)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))
-include $(DEP)
endif
