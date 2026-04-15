-8a
runfile("Step08a_extract_1d_final.py", args="--set EVEN")
runfile("Step08a_extract_1d_final.py", args="--set ODD")

-8b
runfile("step08b_merge_even_odd_final.py")

-8c
runfile("step08c_attach_wavelength_final.py", args="--overwrite")

QC
runfile("qc_step08_extract.py", args="--set EVEN")
runfile("qc_step08_extract.py", args="--set ODD")
QC single slit
runfile("qc_step08_extract.py", args="--set EVEN --slit SLIT018")
