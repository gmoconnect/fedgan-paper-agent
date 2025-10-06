import argparse
import os
import tensorflow as tf

from .utils.data_loader import load_mnist_for_agents
from .agents.agent import Agent
from .server.intermediary import Intermediary
from .utils.visuals import save_images_grid, make_gif, save_losses_csv_png


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--agents', type=int, default=3)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--sync_k', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--noise_dim', type=int, default=100)
    p.add_argument('--outdir', type=str, default='outputs')
    return p.parse_args()


def main():
    args = parse_args()
    # define class splits evenly
    classes = list(range(10))
    splits = []
    per = max(1, len(classes)//args.agents)
    for i in range(args.agents):
        splits.append(classes[i*per:(i+1)*per])
    # last agent gets remaining
    if len(splits) > 0 and sum(len(s) for s in splits) < len(classes):
        splits[-1].extend(classes[len(splits)*per:])

    agents_data = load_mnist_for_agents(splits, batch_size=args.batch_size)

    agents = []
    for i, d in enumerate(agents_data):
        agents.append(Agent(i, d['dataset'], d['size'], noise_dim=args.noise_dim))

    server = Intermediary(agents)

    # fixed seed for evaluation images
    num_examples = 16
    seed = tf.random.normal([num_examples, args.noise_dim])

    os.makedirs(args.outdir, exist_ok=True)
    gif_files = []

    for epoch in range(1, args.epochs+1):
        # local training
        for agent in agents:
            agent.train_one_epoch(seed)

        # sync
        if epoch % args.sync_k == 0:
            server.aggregate()

        # generate images per agent for visualization and save losses
        imgs_dir = os.path.join(args.outdir, f'epoch_{epoch}')
        os.makedirs(imgs_dir, exist_ok=True)
        for agent in agents:
            gen = agent.generator(seed, training=False).numpy()
            out_path = os.path.join(imgs_dir, f'agent_{agent.id}.png')
            save_images_grid(gen, epoch, out_path)
            # save losses
            loss_csv = os.path.join(args.outdir, f'agent_{agent.id}_losses.csv')
            loss_png = os.path.join(args.outdir, f'agent_{agent.id}_losses.png')
            save_losses_csv_png(agent, loss_csv, loss_png)
            if agent.id == 0:
                gif_files.append(out_path)

    # produce gif from agent 0 images
    gif_out = os.path.join(args.outdir, 'training_progress.gif')
    make_gif(gif_files, gif_out)


if __name__ == '__main__':
    main()
