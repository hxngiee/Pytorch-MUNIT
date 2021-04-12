# MUNIT                                               
Multimodal Unsupervised Image-to-Image Translation
        
## Train
    $ python main.py --mode train \
                     --scope munit \
                     --name_data cezanne2photo \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint \
                     --ny_load 178 \
                     --nx_load 178 \
                     --gpu_ids 0                                  

## Test
    $ python main.py --mode test \
                     --scope munit \
                     --name_data cezanne2photo \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoints \
                     --ny_load 178 \
                     --nx_load 178 \
                     --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                     --dir_result ./results
                     --gpu_ids 0
