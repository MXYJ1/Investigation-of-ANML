# Investigation-of-ANML
This uses the code from the public GitHub repository https://github.com/uvm-neurobotics-lab/ANML

To train the original ANML model:
%%shell
python mrcl_classification.py --rln 7 --meta_lr 0.001 --update_lr 0.1 --name mrcl_omniglot --steps 20000 --seed 9 --model_name "Neuromodulation_Model.net" 

To evaluate the meta-test training performance of the original ANML model:
%%shell
python evaluate_classification.py --rln 13 --model Neuromodulation_Model.net --name Omni_test_traj --runs 1 --neuromodulation

To evaluate the meta-test testing performance of the original ANML model:
%%shell
python evaluate_classification.py --rln 13 --model Neuromodulation_Model.net --name Omni_test_traj --runs 1 --neuromodulation --test
