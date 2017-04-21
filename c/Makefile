IDIR =include
# MATLAB_DIR=/usr/local/MATLAB/R2012a/extern/include
CC=c++

CFLAGS=-I$(IDIR)

# compiled module directory
ODIR =build
# Libraries
LDIR =/shared/users/asousa/WIPP/WIPP_stencil/c/lib

LIBS= 
	
	
# output binary directory
BDIR =bin
# source files here
SRC_DIR=src

# Dependencies (header files)
_DEPS = WIPP_stencil.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

sources = \
	ctypes_trial.cpp \
	calc_scattering.cpp \
	fresnel.cpp \
	flux_methods.cpp

sources_full = $(patsubst %,$(SRC_DIR)/%,$(sources))
_OBJ = ${sources:.cpp=.o}	


OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRC_DIR)/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS) -L$(LDIR)

all: libwipp.a

clean:
	${RM} *.o
	${RM} *.mod
	${RM} *.a
	${RM} $(ODIR)/libwipp.a
	${RM} libwipp.so

libwipp.a: ${OBJECTS}
	# Make static library
	ar rc libwipp.a ${OBJECTS}
	cp libwipp.a $(ODIR)

shared:
	${CC} $(CFLAGS) -L$(LDIR) -shared ${sources_full} -o libwipp.so -fPIC