# HRAN(x2) 
#python main.py --model HRAN --scale 2 --patch_size 96 --save HRAN_x2 --reset

# HRAN(x3) 
#python main.py --model HRAN --scale 2 --patch_size 96 --save HRAN_x3 --reset

# HRAN(x4) 
#python main.py --model HRAN --scale 2 --patch_size 96 --save HRAN_x4 --reset


# Test your own images
python main.py --model HRAN --data_test Demo --scale 2  --test_only --save_results

