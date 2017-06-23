NAME := opencvtest

CC := g++ # This is the main compiler
SRCDIR := src
BUILDDIR := build
TARGETDIR := bin
EXECUTABLE := $(NAME)
TARGET := $(TARGETDIR)/$(EXECUTABLE)

SRCEXT := cpp
SOURCES := $(SRCDIR)/$(NAME).$(SRCEXT)
OBJECTS := $(BUILDDIR)/$(NAME).o

# Special OpenCV Flags
OPENCV := $(shell pkg-config --cflags --libs opencv)

CFLAGS := -c -g #-Wall
LIB := -L lib
INC := -I include

$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGETDIR)
	@echo " Linking $<..."; $(CC) $(OPENCV) $^ -o $(TARGET)
	@echo " Executable: $(TARGET)"

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	@mkdir -p $(BUILDDIR)
	@echo " Compiling $<..."; $(CC) $(CFLAGS) $(INC) -o $@ $<

clean:
	@echo " Cleaning $(TARGET)..."; $(RM) -r $(BUILDDIR) $(TARGET)

reset:
	@echo " Resetting..."; make clean; make

look:
	@echo " SOURCES: $(SOURCES)"
	@echo " OBJECTS: $(OBJECTS)"
	@echo " OPENCV: $(OPENCV)"

.PHONY: clean