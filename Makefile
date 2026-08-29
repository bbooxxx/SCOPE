# Non-volatile memory simulator

#target := tsvtest
target := destiny

# define tool chain
CXX := g++
RM := rm -f

# define build options
# compile options
CXXFLAGS := -Wall 
# link options
LDFLAGS :=
# link librarires
LDLIBS :=

OUTDIR := obj

# construct list of .cpp and their corresponding .o and .d files
SRC := main.cpp $(wildcard component/*.cpp)
INC := -I. -Icomponent
DBG :=
OBJ := $(OUTDIR)/main.o $(patsubst component/%.cpp,$(OUTDIR)/%.o,$(wildcard component/*.cpp))
DEP := Makefile.dep

# file disambiguity is achieved via the .PHONY directive
.PHONY : all clean dbg scope scope-requested test-scope

all: CXXFLAGS += -O3 -mtune=native
all: dir $(target)

dbg: DBG += -ggdb -g #-DNVSIM3DDEBUG=1
dbg: dir $(target)

dir:
	mkdir -p $(OUTDIR)

$(target): $(OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) $(target) $(DEP) $(OBJ)

scope: all
	python3 scope.py config/scope_v3.json --json-output results/scope_v3.json

scope-requested: all
	python3 scope.py config/scope_v2_requested.json --json-output results/scope_v2_requested.json

test-scope:
	python3 -m unittest discover -s tests -v

$(OUTDIR)/main.o: main.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

$(OUTDIR)/%.o: component/%.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

depend $(DEP):
	@echo Makefile - creating dependencies for: $(SRC)
	@$(RM) $(DEP)
	@$(CXX) -E -MM $(INC) $(SRC) >> $(DEP)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))
-include $(DEP)
endif
