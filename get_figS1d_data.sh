export gate_type="haar"
export error_type="depol"
export solver_type="scipy"

# debug
export eps_list="[1e-4]"
export c_list="[0.3,0.6,0.9,1.2,1.5]"
export J_list="[1,2,3,4,5,6]"
export n_data=2000
export save_each_J=true
export save_each_c=true

python ./calc_success_rate.py --gate_type $gate_type --error_type $error_type --solver_type $solver_type --eps_list $eps_list --c_list $c_list --n_data $n_data --J_list $J_list --save_each_J $save_each_J --save_each_c $save_each_c