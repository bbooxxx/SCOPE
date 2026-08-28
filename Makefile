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
SRC := $(wildcard *.cpp)
INC := 
DBG :=
OBJ := $(patsubst %.cpp,$(OUTDIR)/%.o,$(notdir $(SRC)))
DEP := Makefile.dep

# file disambiguity is achieved via the .PHONY directive
.PHONY : all clean dbg scope test-scope

all: CXXFLAGS += -O3 -mtune=native
all: dir $(target)

dbg: DBG += -ggdb -g #-DNVSIM3DDEBUG=1
dbg: dir $(target)

dir:
	mkdir -p $(OUTDIR)

$(target): $(OBJ)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

clean:
	$(RM) $(target) $(dep_file) $(OBJ)

scope: all
	python3 scope.py config/scope_example.json --json-output results/scope_example.json

test-scope:
	python3 -m unittest discover -s tests -v

$(OUTDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DBG) $(INC) -c $< -o $@

depend $(DEP):
	@echo Makefile - creating dependencies for: $(SRC)
	@$(RM) $(DEP)
	@$(CXX) -E -MM $(INC) $(SRC) >> $(DEP)

ifeq (,$(findstring clean,$(MAKECMDGOALS)))
-include $(DEP)
endif
