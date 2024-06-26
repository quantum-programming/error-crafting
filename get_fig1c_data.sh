export gate_type="haar"
export error_type="pauli"
export solver_type="scipy"

export eps_list="[1e-4]"
export c_list="[1.1,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0]"
export J_list="[1,2]"
export n_data=10000
export save_each_J=true
export save_each_c=true

# latter half
#export eps_list="[1e-4]"
#export c_list="[0.3,0.6,0.9,1.2,1.5,1.8,2.1,2.4,2.7,3.0]"
#export J_list="[3,4]"
#export n_data=10000
#export save_each_J=true
#export save_each_c=true


python ./calc_success_rate.py --gate_type $gate_type --error_type $error_type --solver_type $solver_type --eps_list $eps_list --c_list $c_list --n_data $n_data --J_list $J_list --save_each_J $save_each_J --save_each_c $save_each_c