universe = vanilla
+AccountingGroup = "group_rutgers.amit"
getenv = true
initialdir = .
+C7OK = "yes"
error =./con_logs/limit_**tag_**mean_$(Process).error
log =./con_logs/limit_**tag_**mean_$(Process).log
output =./con_logs/limit_**tag_**mean_$(Process).out
executable = run_limits.sh
arguments = $(Process) **input_file **length_scale **variance **mean **sigma **rate_uc **mean_err **sigma_err **nwalkers **steps **sig_strength **length_scale_err **variance_err
Notification=never
queue 0000