LIBRARY_DIR=/path/to/itensor2
include $(LIBRARY_DIR)/this_dir.mk
include $(LIBRARY_DIR)/options.mk

################################################################
#Options --------------

HEADERS=

ifdef app
APP=$(app)
else
APP=electronBO
endif


#################################################################

OBJECTS=$(APP).o

#Mappings --------------
GOBJECTS=$(patsubst %,.debug_objs/%, $(OBJECTS))

#Rules ------------------

%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cc $(HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

%.o: %.cpp $(HEADERS)
	$(CCCOM) -c $(CCFLAGS) -o $@ $<

.debug_objs/%.o: %.cpp $(HEADERS)
	$(CCCOM) -c $(CCGFLAGS) -o $@ $<

#Targets -----------------

default: build

build: $(APP)
debug: $(APP)-g

$(APP): $(OBJECTS) 
	$(CCCOM) $(CCFLAGS) $(OBJECTS) -o $(APP) $(LIBFLAGS)

$(APP)-g: mkdebugdir $(GOBJECTS) $(ITENSOR_GLIBS)
	$(CCCOM) $(CCGFLAGS) $(GOBJECTS) -o $(APP)-g $(LIBGFLAGS)

mkdebugdir:
	mkdir -p .debug_objs

clean:
	rm -fr *.o $(APP) $(APP)-g .debug_objs
