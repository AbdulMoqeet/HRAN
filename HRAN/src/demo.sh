# HRAN(x2) 
#python main.py --template HRAN --model HRAN --scale 2 --patch_size 96 --save HRAN_x2 --ext sep_reset

# HRAN(x4) 
#python main.py --template HRAN --model HRAN --scale 4 --patch_size 192 --save HRAN_x4 --ext sep_reset

# HRAN(x8) 
#python main.py --template HRAN --model HRAN --scale 4 --patch_size 384 --save HRAN_x8 --ext sep_reset

## Testing

# HRAN(x2)
python3 main.py --template HRAN --model HRAN --scale 2 --save HRAN_x2 --pre_train ../trained_models/model_best_x2.pt --test_only --save_results

# HRAN(x4)
 python3 main.py --template HRAN --model HRAN --scale 4  --save HRAN_x4 --pre_train ../trained_models/model_best_x4.pt --test_only --save_results

# HRAN(x8)
 python3 main.py --template HRAN --model HRAN --scale 8  --save HRAN_x8 --pre_train ../trained_models/model_best_x8.pt --test_only --save_results


# Test your own images
python main.py --model HRAN --data_test Demo --scale 2  --test_only --save_results

