import networkx as nx
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------
# 1. Create Graph
# ------------------------
G = nx.Graph()
G.add_edges_from([
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
	(1, 4),
	(2, 4),
	(0, 5),
	(3, 5)
])

nodes = list(G.nodes())

# ------------------------
# 2. Biased random walks
# ------------------------
def node2vec_walk(G, start_node, walk_length, p=1, q=1):
    walk = [start_node]
    while len(walk) < walk_length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if len(neighbors) == 0:
            break

        if len(walk) == 1:
            walk.append(random.choice(neighbors))
        else:
            prev = walk[-2]
            probs = []
            for nbr in neighbors:
                if nbr == prev:
                    probs.append(1/p)
                elif G.has_edge(prev, nbr):
                    probs.append(1)
                else:
                    probs.append(1/q)
            probs = torch.tensor(probs, dtype=torch.float)
            probs /= probs.sum()
            walk.append(random.choices(neighbors, weights=probs, k=1)[0])
    return walk

# ------------------------
# 3. Ranodom Walks corpus
# ------------------------
def generate_walks(G, num_walks=10, walk_length=5, p=1, q=1):
    walks = []
    for _ in range(num_walks):
        for node in G.nodes():
            walks.append(node2vec_walk(G, node, walk_length, p, q))
    return walks

walks = generate_walks(G, num_walks=20, walk_length=5, p=1, q=1)

# ------------------------
# 4. Positives pairs (context windows)
# ------------------------
window_size = 2
pairs = []
for walk in walks:
    for i, u in enumerate(walk):
        for j in range(max(0, i - window_size), min(len(walk), i + window_size + 1)):
            if i != j:
                pairs.append((u, walk[j]))

# ------------------------
# 5. Skip-gram with Negative Sampling
# ------------------------
class SGNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, negatives):
        """
        center: (batch,)
        context: (batch,)
        negatives: (batch, k)
        """
        center_emb = self.in_embed(center)               # (batch, d)
        context_emb = self.out_embed(context)           # (batch, d)
        neg_emb = self.out_embed(negatives)             # (batch, k, d)

        # Positive score
        pos_score = torch.sum(center_emb * context_emb, dim=1)        # (batch,)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        # Negative score
        # batch matrix multiplication: (batch, 1, d) x (batch, d, k) -> (batch, 1, k)
        neg_score = torch.bmm(center_emb.unsqueeze(1), neg_emb.transpose(1,2)).squeeze(1)  # (batch, k)
        neg_loss = torch.log(torch.sigmoid(-neg_score)).sum(dim=1)                         # (batch,)

        return -(pos_loss + neg_loss).mean()



# ------------------------
# 6. Training
# ------------------------
vocab_size = len(nodes)
embed_dim = 2
k = 3   # negative samples

model = SGNS(vocab_size, embed_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

pairs_idx = [(nodes.index(u), nodes.index(v)) for u, v in pairs]

for epoch in range(50):
	print("Epoch:", epoch)
	total_loss = 0
	random.shuffle(pairs_idx)
	for u, v in pairs_idx:
		negatives = []
		while len(negatives) < k:
			neg = random.choice(nodes)
			if neg != v:
				negatives.append(nodes.index(neg))
		negatives = torch.tensor([negatives], dtype=torch.long)
		loss = model(torch.tensor([u]), torch.tensor([v]), negatives)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_loss += loss.item()
	if epoch % 10 == 0:
		print(f"Epoch {epoch}, loss {total_loss:.4f}")

# ------------------------
# 7. Embedding final
# ------------------------
embeddings = model.in_embed.weight.data.numpy()
for i, node in enumerate(nodes):
	print(f"Node {node}: {embeddings[i]}")

# ------------------------
# 7. Embedding visualization
# ------------------------

if embed_dim == 2:
	x, y = embeddings[:, 0], embeddings[:, 1]
	plt.figure(figsize=(10, 10))
	plt.scatter(x, y, s=120, c='C0')
	for idx, node in enumerate(nodes):
		plt.annotate(str(node), (x[idx], y[idx]), textcoords="offset points", xytext=(6, 6))
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True, linestyle='--', alpha=0.3)
	plt.xlabel('dim 1')
	plt.ylabel('dim 2')
	plt.title('Node embeddings (2D)')
	plt.show()
elif embed_dim == 3:
	from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111, projection='3d')
	xs, ys, zs = embeddings[:, 0], embeddings[:, 1], embeddings[:, 2]
	ax.scatter(xs, ys, zs, s=120, c='C0')
	for idx, node in enumerate(nodes):
		ax.text(xs[idx], ys[idx], zs[idx], str(node))
	ax.set_xlabel('dim 1')
	ax.set_ylabel('dim 2')
	ax.set_zlabel('dim 3')
	plt.title('Node embeddings (3D)')
	plt.show()

