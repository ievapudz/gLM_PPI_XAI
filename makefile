# Makefile to test the environment.

TEST_DIR = tests

INPUT_TEST_DIR = ${TEST_DIR}/cases

OUTPUT_TEST_DIR = ${TEST_DIR}/outputs

TEST_CASE_SCRIPTS = $(sort $(wildcard ${INPUT_TEST_DIR}/*.sh))
TEST_CASE_OUTPUTS = ${TEST_CASE_SCRIPTS:${INPUT_TEST_DIR}/%.sh=${OUTPUT_TEST_DIR}/%.out}
TEST_CASE_DIFFS   = ${TEST_CASE_SCRIPTS:${INPUT_TEST_DIR}/%.sh=${OUTPUT_TEST_DIR}/%.diff}

PT_FILES = $(wildcard ${OUTPUT_TEST_DIR}/*.pt)

.PHONY: all test check tests checks clean distclean mostlyclean cleanAll

all: tests

.PHONY: display

display:
	@echo ${TEST_CASE_DIFFS}

test tests: ${TEST_CASE_DIFFS}

${OUTPUT_TEST_DIR}/%.diff: ${INPUT_TEST_DIR}/%.sh ${OUTPUT_TEST_DIR}/%.out
	@$< 2>&1 | \
    sed 's/at \(.*\) line [0-9][0-9]*\./at \1 line 999\./' | \
    sed 's/\(.*\) params/999 params/' | \
    sed 's/Epoch \(.*\)/Epoch 999/' | \
    sed 's/Training\(:\|\s\)\(.*\)/Training/' | \
    sed 's/Validation\(:\|\s\)\(.*\)/Validation/' | \
    sed 's/Sanity \(.*\)/Sanity/' | \
    sed 's/\x1b//g' | sed 's/\r//g' | \
	sed 's/wandb sync \(.*\)/wandb sync -/' | diff - $(word 2,$^) | tee $@ 
	@if [ -s $@ ]; then echo "Test $< did not pass:"; cat $@; else echo "Test $< passed."; fi

clean:
	rm -f ${TEST_CASE_DIFFS}
	rm ${PT_FILES}

