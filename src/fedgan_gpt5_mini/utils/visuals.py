import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import pandas as pd

def save_images_grid(images, epoch, out_path, scale=1.0):
    # images: [N, 28,28,1] in [-1,1]
    n = images.shape[0]
    cols = int(np.sqrt(n))
    rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    for i in range(rows*cols):
        ax = axes.flatten()[i]
        ax.axis('off')
        if i < n:
            img = (images[i].squeeze() * 127.5 + 127.5).astype('uint8')
            ax.imshow(img, cmap='gray')
            ax.set_title(f'E{epoch}')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

def make_gif(image_files, out_gif):
    frames = []
    for f in image_files:
        frames.append(imageio.imread(f))
    imageio.mimsave(out_gif, frames, duration=0.5)

def save_losses_csv_png(agent, out_csv, out_png=None):
    # agent: Agent instance with gen_losses and disc_losses lists
    df = pd.DataFrame({'generator_loss': agent.gen_losses, 'discriminator_loss': agent.disc_losses})
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index_label='epoch')
    if out_png:
        plt.figure()
        plt.plot(df['generator_loss'], label='gen')
        plt.plot(df['discriminator_loss'], label='disc')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f'Agent {agent.id} Losses')
        plt.savefig(out_png)
        plt.close()
