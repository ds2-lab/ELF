
# Compiler settings
CXX = g++
CXXFLAGS = -O3 -std=c++17 -pthread

# Directories
SRCDIR = ELF
BUILDDIR = build

# Executables and their corresponding source files
EXECS = elf_pthread de_pthread
SOURCES_elf_pthread = $(SRCDIR)/elf_pthread.cpp
SOURCES_de_pthread = $(SRCDIR)/de_pthread.cpp

# Add prefix to executables for the build directory
TARGETS = $(addprefix $(BUILDDIR)/, $(EXECS))

# Default rule to build all executables
all: $(TARGETS)

# Rule to create each executable
$(BUILDDIR)/elf_pthread:
	$(CXX) $(CXXFLAGS) $(SOURCES_elf_pthread) -o $@

$(BUILDDIR)/de_pthread:
	$(CXX) $(CXXFLAGS) $(SOURCES_de_pthread) -o $@

# Ensure the build directory exists
$(TARGETS): | $(BUILDDIR)

# Rule to create the build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Clean rule
clean:
	rm -f $(TARGETS)

.PHONY: all clean

