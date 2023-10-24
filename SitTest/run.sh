all_sample_num=50
model_name="gpt-3.5-turbo"
num_box=10
for init_flag in True False
do
    for arg_flag in True False
    do
        for func_flag in True False
        do
            for shot_num in {1..2}
            do
                python run.py \
                  --debug True \
                  --num_box ${num_box} \
                  --num_key ${num_box} \
                  --model ${model_name} \
                  --all_sample_num ${all_sample_num} \
                  --with_irreg_func ${func_flag} \
                  --with_irreg_arg ${arg_flag} \
                  --shots_num ${shot_num} \
                  --with_init ${init_flag}
# only uncomment when you have setup the API key and prepared to spend budget
#                python run.py \
#                  --debug False \
#                  --num_box ${num_box} \
#                  --num_key ${num_box} \
#                  --model ${model_name} \
#                  --all_sample_num ${all_sample_num} \
#                  --with_irreg_func ${func_flag} \
#                  --with_irreg_arg ${arg_flag} \
#                  --shots_num ${shot_num} \
#                  --with_init ${init_flag}
            done
        done
    done
done
