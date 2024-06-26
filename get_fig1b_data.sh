export gate_type="haar"
export error_type="pauli"
export solver_type="scipy"

export eps_list="[1e-4]"
export c_list="[0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0]"
export J_list="[1,2,3,4,5,6,7,8,9,10]"
export n_data=300
export save_each_J=true
export save_each_c=true

python ./calc_success_rate.py --gate_type $gate_type --error_type $error_type --solver_type $solver_type --eps_list $eps_list --c_list $c_list --n_data $n_data --J_list $J_list --save_each_J $save_each_J --save_each_c $save_each_c