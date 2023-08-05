from pathlib import Path
import sys
sys.path.append("D:\SourceCodes\PTI-Face")

import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper


current_directory = os.getcwd()
image_dir_name = 'image'

image_name = 'personal_image'
use_multi_id_training = False

global_config.device = 'cuda'
paths_config.e4e = f'{current_directory}/pretrained_models/e4e_ffhq_encode.pt'
paths_config.stylegan2_ada_ffhq = f'{current_directory}/pretrained_models/ffhq.pkl'
paths_config.dlib = f'{current_directory}/pretrained_models/shape_predictor_68_face_landmarks.dat'

paths_config.input_data_id = image_dir_name
paths_config.input_data_path = f'{current_directory}/{image_dir_name}_processed'
paths_config.checkpoints_dir = f'{current_directory}/checkpoints'
paths_config.style_clip_pretrained_mappers = f'{current_directory}/pretrained_models'
hyperparameters.use_locality_regularization = False

def export_updated_pickle(new_G , model_id):
    with open(paths_config.stylegan2_ada_ffhq, "rb") as f:
        d = pickle.load(f)
        old_G = d['G_ema'].cuda()
        old_D = d['D'].eval().requires_grad_(False).cpu()

    tmp = {}
    tmp['G'] = old_G.eval().requires_grad_(False).cpu()
    tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
    tmp['D'] = old_D
    tmp['training_set_kwargs'] = None
    tmp['augment_pip'] = None

    with open(f'{paths_config.checkpoints_dir}/model_stylegan2_custom.pkl', 'wb') as f:
        pickle.dump(tmp, f)

def display_alongside_source_image(images): 
    res = np.concatenate([np.array(image) for image in images], axis=1) 
    return Image.fromarray(res) 

def load_generators(model_id, image_name):
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].cuda()
    
    with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new: 
        new_G = torch.load(f_new).cuda()

    return old_G, new_G 

def plot_syn_images(syn_images): 
    
    for img in syn_images: 
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
        plt.axis('off') 
        resized_image = Image.fromarray(img,mode='RGB').resize((256,256)) 
        display(resized_image) 
        #resized_image.show()
        #plt.imshow(resized_image)
        del img 
        del resized_image 
        torch.cuda.empty_cache()

def run():
    current_directory = os.getcwd()

    model_id = 'face'

    while 1:
        print("1. Preprocess custom image (./raw_images/personal_image.*(jpg or png ...))")
        print("2. Run PTI with custom image.")
        print("3. Visualize result.")
        print("4. Export updated pickle.")
        print("5. Exit.")

        print("\nInput:")
        cmdInput = input()
        try:
            nCommand = int(cmdInput)
        except:
            print("Existing...")
            return
        
        if nCommand >= 5:
            print("Existing...")
            return
        elif nCommand == 1:
            # Preprocess image
            pre_process_images(f"{current_directory}/raw_images")
        elif nCommand == 2:  
            # Run PTI
            model_id = run_PTI(run_name=model_id , use_wandb=False, use_multi_id_training=use_multi_id_training)
        else:  
            # If multi_id_training was used for several images. 
            # You can alter the w_pivot index which is currently configured to 0, and then running the visualization code again. 
            # Using the same generator on different latent codes.
            w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'
            embedding_dir = f'{w_path_dir}/{paths_config.pti_results_keyword}/{image_name}'
            w_pivot = torch.load(f'{embedding_dir}/0.pt')

            generator_type = paths_config.multi_id_model_type if use_multi_id_training else image_name
            old_G, new_G = load_generators(model_id, generator_type)
            old_image = old_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)
            new_image = new_G.synthesis(w_pivot, noise_mode='const', force_fp32 = True)

            #output_image = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
            #output_image = Image.fromarray(output_image,mode='RGB')
            #output_image.show()
            plot_syn_images([old_image, new_image])
            if nCommand == 3:
                latent_editor = LatentEditorWrapper()
                latents_after_edit = latent_editor.get_single_interface_gan_edits(w_pivot, [-2, 2])
                for direction, factor_and_edit in latents_after_edit.items():
                    print(f'Showing {direction} change')
                    for latent in factor_and_edit.values():
                        new_image = new_G.synthesis(latent, noise_mode='const', force_fp32 = True)
                        output_image = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0]
                        output_image = Image.fromarray(output_image,mode='RGB')
                        output_image.show(title=direction)
            elif nCommand == 4:
                export_updated_pickle(new_G, model_id)

if __name__ == '__main__':
    run()
