class Intermediary:
    def __init__(self, agents):
        self.agents = agents

    def aggregate(self):
        # Weighted average of generator and discriminator weights by agent data size
        total = sum(a.data_size for a in self.agents)
        if total == 0:
            return None, None

        # Initialize accumulators
        # Use first agent to get shapes
        first_gw, first_dw = self.agents[0].get_weights()
        gen_weights = [wg * 0.0 for wg in first_gw]
        disc_weights = [wd * 0.0 for wd in first_dw]

        for agent in self.agents:
            gw, dw = agent.get_weights()
            w = float(agent.data_size) / float(total)
            gen_weights = [g_acc + (wg * w) for g_acc, wg in zip(gen_weights, gw)]
            disc_weights = [d_acc + (wd * w) for d_acc, wd in zip(disc_weights, dw)]

        # Set averaged weights to all agents
        for agent in self.agents:
            agent.set_weights(gen_weights, disc_weights)

        return gen_weights, disc_weights
