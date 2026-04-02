from keras import layers
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from skimage.measure import block_reduce
import time
import gc
import keras
import csv

## Adapted From https://github.com/brain-research/self-attention-gan/blob/master/non_local.py ###
    
class SelfAttention3D(layers.Layer):
    def __init__(self, channel_in, cbar=None, **kwargs):
        super().__init__(**kwargs)
        if cbar is None:
            self.cbar = channel_in // 2
        else:
            self.cbar = cbar
        # projections
        self.f_conv = layers.Conv3D(self.cbar, 1, padding="same")
        self.g_conv = layers.Conv3D(self.cbar, 1, padding="same")
        self.h_conv = layers.Conv3D(self.cbar, 1, padding="same")
        self.o_conv = layers.Conv3D(channel_in, 1, padding="same")
        # trainable scalar
        self.gamma = self.add_weight(
            name="gamma", shape=(), initializer="zeros", trainable=True )

    def call(self, x):
        f = self.f_conv(x); g = self.g_conv(x); h = self.h_conv(x)
        # flatten dynamically
        f_flat = tf.reshape(f, [-1, tf.shape(f)[1]*tf.shape(f)[2]*tf.shape(f)[3], self.cbar])
        g_flat = tf.reshape(g, [-1, tf.shape(g)[1]*tf.shape(g)[2]*tf.shape(g)[3], self.cbar])
        h_flat = tf.reshape(h, [-1, tf.shape(h)[1]*tf.shape(h)[2]*tf.shape(h)[3], self.cbar])
        # attention
        s = tf.matmul(g_flat, f_flat, transpose_b=True)
        beta = tf.nn.softmax(s)
        bh = tf.matmul(beta, h_flat)
        bh = tf.reshape(bh, [-1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3], self.cbar])
        o = self.o_conv(bh)
        return x + self.gamma * o
    
def load_data(path, erange, interaction_model, start_load=0, end_load=None, divide = 1):
    # path: the parent directroy containig subdirectories with data and metadata e.g. "../../DataCorsika7/img_data/76310_1e6_2e7/"
    # erange: the energy range of interest, e.g. "1e5-1e6" "1e6-2e7" "2e7-1e8"
    # interaction_model: the interaction model of interest, e.g. "76310"

    start_time = time.time()
    def print_with_time(message):
        elapsed = time.time() - start_time
        print(f"{message} [{elapsed:.2f} s]")

    dirs = os.listdir(path)
    dirs = [entry for entry in dirs if os.path.isdir(os.path.join(path, entry))]
    data_x = []
    meta = []

    ################################ Load all data and metadata files ################################
    for entry in dirs[start_load:end_load]:
        data_x.append(np.load(f"{path}{entry}/mu_p_data_{entry}.npy"))
        meta.append(pd.read_csv(f"{path}{entry}/mu_p_metadata_{entry}.csv"))
    print_with_time("loaded")

    meta_all = pd.concat(meta, ignore_index=True)
    data_x = np.concatenate(data_x, axis=0)
    for i in range(10):
        gc.collect()
    print_with_time("concatenated data")

    ############################################ DownSample  #########################################
    # Reduce along pz dimension
    z = 13; y = 34-2*z; x = z-2 # tested for energy range 1e6-2e7
    data_x = block_reduce(data_x,
                        block_size=(1, 2, 2, 1, 2),
                        func=np.sum)
    print_with_time("block reduced data - 1")

    for i in range(10):
        gc.collect()
    un = block_reduce(data_x[:,:,:,:y,:],
                        block_size=(1, 1, 1, 2, 1),
                        func=np.sum)
    print_with_time("block reduced data - 2")
    du = data_x[:,:,:,y:32-z,:]
    tre = data_x[:,:,:,32-z:,:].sum(axis=3,keepdims=True)
    data_x = np.concatenate((un,du,tre),axis=3)
    print_with_time("concatenated un, du, tre")
    # now I have in each bin the number of particles falling in that bin,
    # I apply a transformation to make the range of values more suitable for the network to learn
    #  (e.g. log10(1+particles) or log10(1+particles)/max(log10(1+particles)) or other transformations)

    # TODO modify accordingly to different interpretation of bins
    data_x = np.log10(data_x+1)
    ##### For tanh activation as last layer activation ######
    # shift = (data_x.max() + data_x.min())/2
    # data_x = data_x - shift
    # normal = np.max([-data_x.min(),data_x.max()])
    # data_x = data_x / normal
    
    ##### For sigmoid activation as last layer activation #####
    normal = np.max(data_x)
    data_x = (data_x/normal) / divide
    print_with_time("Scaled data")

    ############################################# Metadata  ##########################################
    labels = meta_all[["Primary_energy","Starting_height"]].to_numpy()

    # General statistics on all Muons
    dic_erange= {"1e5-1e6": (10**5, 10**6),
            "1e6-2e7": (10**6, 2*10**7),         
            "2e7-1e8": (2*10**7, 10**8)}

    pat = "../../DataCorsika7/statistiche_complete/muoni/mu_p_76310_a"
    cosini = [chr(i+97)+".csv" for i in range(12)]
    datiframes = [pd.read_csv(pat+ini) for ini in cosini]
    all_mup76310 =  pd.concat(datiframes, ignore_index=True)
    
    # Provide altitude (Starting_height) and energy (Primary_energy) in a more convenient form for the network to learn
    # (e.g. normal distribution for altitude and uniform distribution in [-2,2] for energy)
    # Useful for a possible future work: Build conditioned GANs to include these info when generating new samples
    SH = all_mup76310.Starting_height 
    labels[:,1] = ((np.log10(labels[:,1])-np.log10(SH).mean())/np.log10(SH).std()) # normal distribution for altitude


    lg10_emin = np.log10(dic_erange[erange][0])
    lg10_emax = np.log10(dic_erange[erange][1])
    labels[:,0] = (np.log10(labels[:,0])-(lg10_emin/2+lg10_emax/2)) / (lg10_emax-lg10_emin)*4 # uniform distribution in [-2,2] for energy
    
    
    ############################################# Details about bins (save in a txt file the details of the binning scheme)  ##########################################
     # statistic of the interaction model considered: (pxmax, pymax, logrmax, logpzmax) - used to retrieve ranges of imeges
    path_statistics = "../../DataCorsika7/statistiche_complete/muoni"
    sts =  pd.read_csv(f"{path_statistics}/Mup_Ranges_{interaction_model}.csv")                 
    factr = 1.5
    
    abspxymax = sts[sts["energy"]=="e"+erange]["pxpy80"].values[0]
    logpzmax = sts[sts["energy"]=="e"+erange]["logpzmx"].values[0]
    rm = 10**sts[sts["energy"]=="e"+erange]["logr90"].values[0]
    
    # TODO modify accordingly to different interpretation of bins
    stringa = f"energy range: {erange}, interaction model: {interaction_model}"
    stringa += f"\n\n{"#"*10} DETAILS ABOUT DATA {"#"*10}"
    stringa += f"\npxpy {data_x.shape[1]} bins: np.linspace(-{abspxymax}, {abspxymax}, 32 + 1), grouped 2 by 2"
    stringa += f"\nr {data_x.shape[4]} bins: [{rm:.2f}/np.sqrt({factr})**_ for _ in range(16)]+[0]; binr.sort(), grouped 2 by 2"
    stringa += f"\nlog_pz {data_x.shape[3]} bins: np.linspace(0, {logpzmax}, 32 + 1)"
    stringa += f"\nz_bins: merged first {y}; summed last {z}; the other {x} are unchanged\n"
    stringa += f"In each bin we have [log10(PARTICLES+1) /{normal:.4f}] /{divide:.4f} range in [0,1/{divide:.4f}]  " # TODO modify accordingly to different interpretation of bins
    
    
    stringa += f"\n\n{"#"*10} DETAILS ABOUT MEATADATA {"#"*10}"
    stringa += f"\nPrimary_energy uniform distribution in [-2,2] for energy scaled as:" 
    stringa += f"\n(np.log10(ENERGY)-(np.log10({dic_erange[erange][1]})+np.log10({dic_erange[erange][0]}))/2) / (np.log10({dic_erange[erange][1]})-np.log10({dic_erange[erange][0]}))*4"
    stringa += f"\nStarting_height normal distribution scaled as:\n"
    stringa += f"((np.log10(ALTITUDE)-{np.log10(SH).mean():.4f})/{np.log10(SH).std():.4f})"

    return data_x, labels, stringa

class WGAN_GP(keras.Model):
    def __init__(
        self,
        critic,
        generator,
        latent_dim,
        critic_extra_steps=3,
        gp_weight=10.0,
    ):
        super().__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.c_steps = critic_extra_steps
        self.gp_weight = gp_weight


    def compile(self, c_optimizer, g_optimizer, c_loss_fn_wass, g_loss_fn):
        super().compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn_wass = c_loss_fn_wass
        self.g_loss_fn = g_loss_fn


    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the critic loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            # Calculate the gradients w.r.t to this interpolated image.
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True) 

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3, 4]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, data):

        real_images = data
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]

        # Train critic
        for _ in range(self.c_steps):
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim))


            with tf.GradientTape() as tape:

                fake_images = self.generator(random_latent_vectors, training=True) 

                # apply critic
                real_logits = self.critic(real_images, training=True) 
                fake_logits = self.critic(fake_images, training=True) 

                # gp (interpolate between same labels)
                gp_images = self.generator(random_latent_vectors, training=True)  

                # compute losses
                self.gp = self.gradient_penalty(batch_size, real_images, gp_images)

                self.d_cost_wass = self.c_loss_fn_wass(real_out=real_logits, fake_out=fake_logits)
                
                self.d_loss = self.d_cost_wass + self.gp_weight*self.gp

            
            d_gradient = tape.gradient(self.d_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(
                zip(d_gradient, self.critic.trainable_variables) )


        # Train generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True) 
            # Get the critic logits for fake images
            gen_img_logits = self.critic(generated_images, training=True)   

            # losses
            self.g_cost_wass = self.g_loss_fn(gen_img_logits) 

            self.g_loss = self.g_cost_wass

        gen_gradient = tape.gradient(self.g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables))
        
        return {"d_loss": self.d_loss, "d_cost_wass": self.d_cost_wass,
                "gradient_penality":self.gp, 
                "g_loss": self.g_loss, "g_cost_wass":self.g_cost_wass}
    
def build_models_WGAN(latent_dim = 64, height=16, width=16, depth=16, channels=8):
    ###################### GENERATOR ######################
    gen_input_latent = keras.Input(shape=(latent_dim,))
    x = layers.Dense(4 * 4 * 4,activation="relu")(gen_input_latent)
    x = layers.Reshape((4, 4, 4, 1))(x)
    x = layers.Conv3D(16, 4, activation="relu", padding="same")(x)

    x = layers.Conv3DTranspose(32, 6, strides=2,activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(32, 6, activation="relu", padding="same")(x)
    x = SelfAttention3D(32)(x)           

    x = layers.Conv3DTranspose(32, 8, 2,activation="relu", padding="same")(x)
    x = layers.Conv3D(32, 3,activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(8, 3,activation="relu", padding="same")(x)
    x = layers.Conv3D(channels, 3, activation='sigmoid', padding="same")(x);  

    generator_WGAN = keras.models.Model(gen_input_latent, x,name="generator")

    ###################### CRITIC ######################
    crit_input_img = layers.Input(shape=(height, width, depth, channels))
    y = layers.Conv3D(32, 4, strides=2, activation="leaky_relu", padding="same")(crit_input_img)

    y = layers.Conv3D(64, 3, activation="leaky_relu", padding="same")(y)
    y = SelfAttention3D(64)(y)

    y = layers.Conv3D(64, 3, activation="leaky_relu", padding="same")(y)
    y = layers.Dropout(0.25)(y)

    y = layers.Conv3D(32, 3, activation="leaky_relu", padding="same")(y)
    y = layers.LayerNormalization(axis=[1,2])(y)

    y = layers.Dropout(0.25)(y)

    y = layers.Conv3D(8, 3, activation="leaky_relu", padding="same")(y)
    y = layers.Flatten()(y)

    wass_out = layers.Dense(64,activation="leaky_relu")(y)
    wass_out = layers.Dense(1)(wass_out)

    critic_WGAN = keras.models.Model(crit_input_img, wass_out, name="critic")

   
    return generator_WGAN, critic_WGAN

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, latent_dim=64, path="../../GitHubFolders/GAN_GAIAS2/GAN/WGAN/", tag="WGAN_GP_corsika_NotConditioned_exceptfirst4_weigdecay_bigkernel", batch_monitor=25):
        self.latent_dim = latent_dim
        self.save_path = path+tag
        self.tag= tag
        self.batch_monitor=batch_monitor
        self.file_losses_path = f"{self.save_path}/_{self.tag}_losses_dataframe.csv"
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # save models and generated images every 30 epochs and for the first 20 epochs
        if epoch <20 or epoch %30 == 0:
            random_latent_vectors = tf.random.normal(shape=(10, self.latent_dim))
            generated_images = self.model.generator(random_latent_vectors)

            for i in range(len(generated_images)):
                img = generated_images[i].numpy()
                imgpxpy = img.sum(axis=(2,3), keepdims=True)[:,:,:,0]                               # 16 16 1  1 
                imgpxpz = img.sum(axis=(1,3), keepdims=True)[:,0,:,:]                               # 16 1  16 1
                imgpxr = img.sum(axis=(1,2), keepdims=True).reshape(img.shape[0],img.shape[3],1)    # 16 1  1  8
                imgpzr = img.sum(axis=(0,1), keepdims=True).reshape(img.shape[2],img.shape[3],1)    # 1  1  16 8 
                imgs = [imgpxpy,imgpxpz,imgpxr,imgpzr]
                lbls = ["pxpy","pxpz","pxr","pzr"]
                for k in range(len(imgs)):
                    imga = keras.utils.array_to_img(imgs[k])
                    imga.save(f"{self.save_path}/{self.tag}_generated_epoch_{epoch}_img_{i}_{lbls[k]}.png".format(i=i, epoch=epoch))
        
            self.model.generator.save(f"{self.save_path}/generator_{self.tag}_{epoch}.keras")
            self.model.critic.save(f"{self.save_path}/critic_{self.tag}_{epoch}.keras")
        
    def on_train_begin(self, logs=None):
        self.file_opened = open(self.file_losses_path,"w")
        self.csv_writer = csv.writer(self.file_opened)
        stringa = "epoch,batch_iteration,"
        stringa += "d_loss,d_cost_wass,gradient_penality,"
        stringa += "g_loss,g_cost_wass"
        self.csv_writer.writerow(stringa.split(","))
    
    def on_train_end(self, logs=None):
        try:
            if self.file_opened is not None:
                self.file_opened.close()
        except Exception as e:
            print(f"Error closing file: {e}")

    def on_train_batch_end(self, batch, logs=None):
        # save losses every batch_monitor iterations
        if batch % self.batch_monitor == self.batch_monitor-1:
            if logs is None:
                logs = {} # logs now contains the numerical values
            list_losses_temp = []
            stringa = "d_loss,d_cost_wass,gradient_penality,"
            stringa += "g_loss,g_cost_wass"
            keys = stringa.split(",")
            for key in keys:
                list_losses_temp.append(logs[key]) # create list [lossd, lossg, lambd...]
            self.csv_writer.writerow([self.current_epoch, batch]+list_losses_temp)
            self.file_opened.flush()
            os.fsync(self.file_opened.fileno())


