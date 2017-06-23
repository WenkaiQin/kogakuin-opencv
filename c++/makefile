
# makefile

NAME := 1cameradepthbyH

CC := g++ # This is the main compiler
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin
TARGET := $(TARGETDIR)/$(NAME)

SRCEXT := cpp
SOURCES := $(SRCDIR)/$(NAME).$(SRCEXT)
OBJECTS := $(BUILDDIR)/$(NAME).o

# Special OpenCV Flags
OPENCV := $(shell pkg-config --cflags --libs opencv)

CFLAGS := -c -g #-std=c++11 #-Wall
LIB := -L lib
INC := -I include

$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGETDIR)
	@echo " Linking $<..."; $(CC) $(OPENCV) $^ -o $(TARGET)
	@echo " Build complete."

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " Compiling $<..."; $(CC) $(CFLAGS) $(INC) -o $@ $<

clean:
	@echo " Cleaning $(TARGET)..."; $(RM) -r $(BUILDDIR) $(TARGET)

purge:
	@echo " Purging $(TARGETDIR)..."; $(RM) -r $(BUILDDIR) $(TARGETDIR)

reset:
	@echo " Resetting..."; make clean; make;

run:
	@echo " Running $(TARGET)..."; ./$(TARGET)

.PHONY: clean