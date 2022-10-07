CC=g++ -O3
LD=g++ -O3
CFLAGS=-O3 -pedantic 
OPT= -g -G
RM=/bin/rm -f

%.o: %.cpp parameter.h
	 $(CC) -c $(CFLAGS) $<
	 
ALL: anneal

anneal: anneal.o
		$(LD) $(LDFLAGS) -o anneal anneal.o -lgsl -lgslcblas -lm
