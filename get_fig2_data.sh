export gate_type="haar"
export error_type="depol"

# run
export eps_list="[1e-4]"
export c_list="[5.0]"
export J_list="[10]"
export n_data=3000

python ./calc_error_channel_profile.py --z_thres $z_thres --gate_type $gate_type --error_type $error_type --eps $eps --c $c --n_data $n_data --J $J
