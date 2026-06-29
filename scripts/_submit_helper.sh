#!/bin/bash
export PAGER=cat
export LESS=
unset LESSOPEN
sbatch /home/moeg/scorecards4extremes/submit_vtb_test.sh > /home/moeg/scorecards4extremes/_jobid.txt 2>&1
