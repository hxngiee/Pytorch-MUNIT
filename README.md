# MUNIT                                               
Multimodal Unsupervised Image-to-Image Translation
        
## Train
    $ python main.py --mode train \
                     --scope [scope name] \
                     --name_data [data name] \
                     --dir_data [data directory] \
                     --dir_log [log directory] \
                     --dir_checkpoint [checkpoint directory] \
                     --ny_load [size y of center crop] \
                     --nx_load [size x of center crop] \
                     --selected_attrs [attributes type]
                     --gpu_ids [gpu id; '-1': no gpu, '0, 1, ..., N-1': gpus]  
---
    $ python main.py --mode train \
                     --scope stargan \
                     --name_data celeba \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint \
                     --ny_load 178 \
                     --nx_load 178 \
                     --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young
                     --gpu_ids 0                                  
---
    $ python main.py --mode train \
                     --scope stargan \
                     --name_data rafd \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoint \
                     --ny_load 640 \
                     --nx_load 640 \
                     --selected_attrs angry contemptuous disgusted fearful happy neutral sad surprised
                     --gpu_ids 0

* Set **[scope name]** uniquely.
* Hyperparameters were written to **arg.txt** under the **[log directory]**.
* To understand hierarchy of directories based on their arguments, see **directories structure** below. 


## Test
    $ python main.py --mode test \
                     --scope [scope name] \
                     --name_data [data name] \
                     --dir_data [data directory] \
                     --dir_log [log directory] \
                     --dir_checkpoint [checkpoint directory] \
                     --ny_load [size y of center crop] \
                     --nx_load [size x of center crop] \
                     --selected_attrs [attributes type] \
                     --dir_result [result directory]       
                     --gpu_ids [gpu id; '-1': no gpu, '0, 1, ..., N-1': gpus]              
---
    $ python main.py --mode test \
                     --scope pix2pix \
                     --name_data celeba \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoints \
                     --ny_load 178 \
                     --nx_load 178 \
                     --selected_attrs Black_Hair Blond_Hair Brown_Hair Male Young \
                     --dir_result ./results
                     --gpu_ids 0
---
    $ python main.py --mode test \
                     --scope pix2pix \
                     --name_data rafd \
                     --dir_data ./datasets \
                     --dir_log ./log \
                     --dir_checkpoint ./checkpoints \
                     --ny_load 640 \
                     --nx_load 640 \
                     --selected_attrs angry contemptuous disgusted fearful happy neutral sad surprised \
                     --dir_result ./results
                     --gpu_ids 0
                     
* To test using trained network, set **[scope name]** defined in the **train** phase.
* Generated images are saved in the **images** subfolder along with **[result directory]** folder.
* **index.html** is also generated to display the generated images.  
