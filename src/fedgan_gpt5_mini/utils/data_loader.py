import tensorflow as tf

def load_mnist_for_agents(agent_splits, batch_size=32):
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # normalize to [-1, 1]

    agents_data = []
    for classes in agent_splits:
        mask = False
        for c in classes:
            mask = mask | (train_labels == c)
        imgs = train_images[mask]
        ds = tf.data.Dataset.from_tensor_slices(imgs).shuffle(10000).batch(batch_size)
        agents_data.append({'dataset': ds, 'size': imgs.shape[0]})

    return agents_data
