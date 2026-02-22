#!/usr/bin/awk -f
# This script merges multiple CSV files into one.
# Usage: awk -f merge.awk /path/to/data/* > merged.csv

FNR == 1 { next }   # skip header lines if they exist

{
    fname = FILENAME
    sub(/^.*\//, "", fname)
    sub(/\.dat$/, "", fname)
    gsub(/\r/, "")
    print $1 "," $2  fname
}
