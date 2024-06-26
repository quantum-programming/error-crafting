
export gate_type="haar"
export error_type="depol"
export solver_type="scipy"

# run
export eps_list="[1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]"
export c_list="[2.0,3.0,5.0,7.0]"
export J_list="[1,3]"
export n_data=200
export save_each_eps=0

python ./calc_dnorm_scaling.py --gate_type $gate_type --error_type $error_type --solver_type $solver_type --eps_list $eps_list --c_list $c_list --n_data $n_data --J_list $J_list --save_each_eps $save_each_eps