# HRAN(x2) 
#python main.py --template HRA --model HRAN --scale 2 --patch_size 96 --save HRAN_x2 --ext sep_reset

# HRAN(x3) 
#python main.py --template HRAN --model HRAN --scale 3 --patch_size 144 --save HRAN_x3 --ext sep_reset

# HRAN(x4) 
#python main.py --template HRAN --model HRAN --scale 4 --patch_size 192 --save HRAN_x4 --ext sep_reset


# Test your own images
python main.py --model HRAN --data_test Demo --scale 2  --test_only --save_results

