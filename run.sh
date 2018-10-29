#!/usr/bin/env bash
# Grupo 12 | Ruben Anagua, 78050 | Ana Luisa Santo, 79758

# SCRIPT IS EXPECTED TO RUN FROM THE SAME DIRECTORY AS ALL OTHER FILES

# Download and extract SRILM from http://www.speech.sri.com/projects/srilm/download.html
# Modify top-level Makefile. Uncomment $SRILM and point it to correct installation path
# Make with `$ make World` and ngram-count will be in a subdir of bin/, in the extracted files
NGRAM_COUNT_EXECUTABLE=../srilm/bin/i686-m64/ngram-count  # change me to correct path!!!

# Generate .final files
python3 out_to_final.py entrava entravaAnotado.out
python3 out_to_final.py fores foresAnotado.out

# Generate .arpa files
$NGRAM_COUNT_EXECUTABLE -text entravaAnotado.final -order 2 -addsmooth 0 -lm entrava.arpa
$NGRAM_COUNT_EXECUTABLE -text foresAnotado.final -order 2 -addsmooth 0 -lm fores.arpa

# Generate Laplace smoothed .arpa files - regular ones are used by 2.py, as it applies Laplace smoothing itself
$NGRAM_COUNT_EXECUTABLE -text entravaAnotado.final -order 2 -addsmooth 0.1 -lm entravaAlisado.arpa
$NGRAM_COUNT_EXECUTABLE -text foresAnotado.final -order 2 -addsmooth 0.1 -lm foresAlisado.arpa

# Run main program
python3 2.py entrava.arpa entravaParametrizacao.txt entravaFrases.txt | tee entravaResultado.txt
python3 2.py fores.arpa foresParametrizacao.txt foresFrases.txt | tee foresResultado.txt

# NOTE: Comments on obtained results (task 3) are on viabilidade.txt
