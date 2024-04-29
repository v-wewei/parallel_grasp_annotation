Follow the following tips for usage:

Firstly, generate the mujoco_asset files used for grasp simulation in MuJoCo:

1. cd scripts/ && python write_obj_xml.py

Secondly, generate initial grasp annotation following the format as GraspNetAPI:

2. cd scripts/ && python grasp_annotation.py

Thirdly, sample high-quality grasps and calculate the in-hand grasp rotation mask:

3. cd scripts/ && python grasp_sampling.py

Finally, get the grasp table with the following command:

4. cd utils/ && python calculate_grasp_table.py



