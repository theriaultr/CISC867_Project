import os

#os.system('CUDA_VISIBLE_DEVICES="0" python3 vae_main.py -mt vae -ol mRNA@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd ember_libfm_200115 -sm')
os.environ["CUDA_VISIBLE_DEVICES"] ="0"
# os.system('CUDA_VISIBLE_DEVICES="0" python3 vae_main.py -mt vae -ol mRNA@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd toyforVAE -sm')
# os.system('python vae_main.py -mt vae -ol mRNA@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd toyforVAE -sm')

# os.system('python vae_main.py -mt vae -ol mrna@ -hn 4096 -lr 0.001 -dr 0 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 500 -af Tanh -mo Adam -bs 0 -vd LIHC_VAE -sm')

#run with 1000 max_epochs dr = 0.5
os.system('python vae_main.py -mt vae -ol mrna@ -hn 4096 -lr 0.1 -dr 0.1 -wd 1e-5 -dv cuda -cd 0 -sn vae_pretrained -fc None -mx 20000 -af Tanh -mo SGD -bs 0 -vd LIHC_VAE -sm') #changed to SGD from Adam
