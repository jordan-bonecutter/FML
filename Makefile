#
#
#
#

COMPILER = gcc
FLAGS    = -Wall -ansi -std=c99
DEBUG    = -O0 -g -DDEBUG
CC       = $(COMPILER) $(FLAGS) $(DEBUG)
CO       = $(CC) -c

fml_net.o: fml_net.c
	$(CO) fml_net.c -o fml_net.o

clean:
	rm -rf *.o

