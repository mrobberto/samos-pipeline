-07d
runfile("pipeline/step07_wavecal/step07d_find_line_shifts.py", args="--set EVEN")runfile("pipeline/step07_wavecal/step07d_find_line_shifts.py", args="--set ODD")QC:runfile("qc/qc07d_arc_alignment.py", args="--set EVEN --plot-all")runfile("qc/qc07d_arc_alignment.py", args="--set ODD --plot-all")

-07f
runfile("pipeline/step07_wavecal/step07f_build_master_arc.py", args="--include EVEN,ODD --write-all-csv")
QC
runfile("qc/qc07f_master_arc.py", args="--save-prefix ../../reduced/07_wavecal/qc07f")