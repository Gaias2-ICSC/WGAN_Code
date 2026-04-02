# nohup ../../GitHubFolders/GAN_GAIAS2/.venv/bin/python ../../GitHubFolders/GAN_GAIAS2/GAN/WGAN_GP_ForCorsika.py &> ../../Desktop/WGAN_GP_ForCorsika_Out_2025_09_13_10_30.txt

import keras
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import subprocess
import numpy as np
import subprocess

from _GAN_utils import load_data, WGAN_GP, build_models_WGAN, GANMonitor

path = "../../GitHubFolders/GAN_GAIAS2/GAN/WGAN/"

tag = "WGAN_GP_corsika_NotConditioned_exceptfirst4_weigdecay_bigkernel"
script_name = "../../GitHubFolders/GAN_GAIAS2/GAN/WGAN_GP_ForCorsika.py"
UTILS_NAME = "../../GitHubFolders/GAN_GAIAS2/GAN/_GAN_utils.py"


###################################################################
# Hyperparameters 
###################################################################
epochs = 100*40
critic_extra_steps=3
latent_dim = 64
BATCH_SIZE = 512
batch_monitor=25

###################################################################
# Select GPU with lowest memory usage
###################################################################
try:
    def get_gpu_memory_usage():
        command = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits"
        memory_usage = subprocess.check_output(command, shell=True).decode("utf-8").strip().split("\n")
        memory_usage = [int(memory) for memory in memory_usage]
        return memory_usage
    mem_usage = get_gpu_memory_usage()
    # Allow memory growth to prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Specify which GPU to use for this script
    # For example, to use GPU 1: tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_visible_devices(gpus[np.argmin(mem_usage)], 'GPU')
except Exception as e:
    print(f"No device selected tf: {e} \n")


###################################################################
# Track time / Backup scripts
###################################################################
now = datetime.now()
formatted_time = now.strftime("%Y-%m-%d_%H_%M_%S")
tag = formatted_time + "_" + tag

os.mkdir(path+tag)
with open(script_name, "r") as f:
    script_content = f.read()
with open(path+tag+'/_Backup_script.txt', "w") as f:
    f.write(script_content)

with open(UTILS_NAME, "r") as fu:
    script_content = fu.read()
with open(path+tag+'/_Backup_script.txt', "a") as f:
    f.write("\n\n\n"+"#"*45+"GAN_utils_Details"+"#"*45+"\n"+script_content)


###################################################################
# Load Data / save details about data and bins
###################################################################
x_train, labels, network_details_stringa = load_data("../../DataCorsika7/img_data/76310_1e6_2e7/", "1e6-2e7", "76310", start_load=4, end_load=None)

with open(path+tag+'/_Network_output_Details.txt', "w") as nf:
    nf.write(network_details_stringa)

###################################################################
# "Image" dimensions and channels (3D "images" with 8 channels)
###################################################################
height = 16
width = 16
depth = 16
channels = 8

###################################################################
# Build WGAN-GP model and compile it with optimizers and loss functions
###################################################################

generator_WGAN, critic_WGAN = build_models_WGAN(latent_dim=latent_dim)
             
cbk = GANMonitor(latent_dim=latent_dim, tag=tag, batch_monitor=batch_monitor)

generator_optimizer = keras.optimizers.Adam(learning_rate=0.00005,  beta_1=0.3, beta_2=0.999, epsilon=0.001)
critic_optimizer = keras.optimizers.RMSprop(learning_rate=0.0002, momentum= 0.3, epsilon=0.001)

def critic_W_loss(real_out, fake_out):
    real_loss = tf.reduce_mean(real_out)
    fake_loss = tf.reduce_mean(fake_out)
    return fake_loss - real_loss

def generator_W_loss(fake_img):
    return -tf.reduce_mean(fake_img)

WGAN_istance = WGAN_GP(
    critic=critic_WGAN,
    generator=generator_WGAN,
    latent_dim=latent_dim,
    critic_extra_steps=critic_extra_steps
)

WGAN_istance.compile(
    c_optimizer=critic_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_W_loss,
    c_loss_fn_wass=critic_W_loss
)

###################################################################
# Train the WGAN
###################################################################
WGAN_istance.fit(x_train, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])  
tf.keras.backend.clear_session()

