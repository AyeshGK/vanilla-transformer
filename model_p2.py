import torch
import torch.nn as nn
import predictive_coding as pc

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embed_dim,
                 n_blocks,
                 n_heads,
                 ff_hid_dim,
                 max_length,
                 dropout,
                 device):
        super().__init__()
        self.encoder = Encoder(src_vocab_size,
                               embed_dim,
                               n_blocks,
                               n_heads,
                               ff_hid_dim,
                               max_length,
                               dropout,
                               device)
        self.decoder = Decoder(trg_vocab_size,
                               embed_dim,
                               n_blocks,
                               n_heads,
                               ff_hid_dim,
                               max_length,
                               dropout,
                               device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).bool().to(self.device) & trg_pad_mask
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.src_mask(src)
        trg_mask = self.trg_mask(trg)
        encoded = self.encoder(src, src_mask)
        decoded = self.decoder(trg, encoded, trg_mask, src_mask)
        return decoded


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout):
        super().__init__()
        self.head_dim = embed_dim // n_heads
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5

        self.keys = nn.Linear(embed_dim, embed_dim)
        self.queries = nn.Linear(embed_dim, embed_dim)
        self.values = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        N = q.size(0)          # batch_size
        Q = self.queries(q)    # shape: [N, query_len, embed_dim]
        K = self.keys(k)       # shape: [N, key_len, embed_dim]
        V = self.values(v)     # shape: [N, value_len, embed_dim]

        Q = Q.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, query_len, head_dim]
        K = K.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, key_len, head_dim]
        V = V.view(N, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # shape: [N, n_heads, value_len, head_dim]

        energy = (Q @ K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e20)

        attention = energy.softmax(-1)           # shape: [N, n_heads, query_len, key_len]
        x = self.dropout(attention) @ V          # shape: [N, n_heads, query_len, key_len]
        x = x.permute(0, 2, 1, 3).contiguous()   # shape: [N, query_len, n_heads, head_dim]
        x = x.view(N, -1, self.embed_dim)        # shape: [N, query_len, embed_dim]
        x = self.proj(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        # self.pcl = pc.PCLayer()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, mask):
        attention = self.attention(src, src, src, mask)
        x = self.norm1(attention + self.dropout(src))
        # pc = self.pcl(x)
        out = self.mlp(x)
        out = self.norm2(out + self.dropout(x))
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_blocks, n_heads, ff_hid_dim, max_length, dropout, device):
        super().__init__()
        self.device = device
        self.scale = embed_dim ** 0.5
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_length, embed_dim)
        self.blocks = nn.ModuleList([EncoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask):
        N, seq_len = src.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        pos_embeddings = self.pos_emb(positions)
        tok_embeddings = self.tok_emb(src) * self.scale
        out = self.dropout(pos_embeddings + tok_embeddings)

        for block in self.blocks:
            out = block(out, mask)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, ff_hid_dim, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_dim, n_heads, dropout)   # decoder self-attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.joint_attention = MultiHeadAttention(embed_dim, n_heads, dropout)  # encoder-decoder attention
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hid_dim, embed_dim)
        )
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        trg_attention = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(trg_attention))
        joint_attention = self.joint_attention(trg, src, src, src_mask)
        trg = self.norm2(trg + self.dropout(joint_attention))
        out = self.mlp(trg)
        out = self.norm3(trg + self.dropout(out))
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_blocks, n_heads, ff_hid_dim, max_length, dropout, device):
        super().__init__()
        self.device = device
        self.scale = embed_dim ** 0.5
        self.tok_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_length, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderLayer(embed_dim, n_heads, ff_hid_dim, dropout)] * n_blocks)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, trg, src, trg_mask, src_mask):
        N, trg_len = trg.shape
        positions = torch.arange(0, trg_len).expand(N, trg_len).to(self.device)
        pos_embeddings = self.pos_embedding(positions)
        tok_embeddings = self.tok_embedding(trg) * self.scale
        trg = self.dropout(pos_embeddings + tok_embeddings)

        for block in self.blocks:
            trg = block(trg, src, trg_mask, src_mask)

        output = self.fc(trg)
        return output


config={
	'PCTrainer_kwargs': {
		'update_x_at': "all",
		'optimizer_x_fn': "SGD",
		'optimizer_x_kwargs': {
			'lr': 0.5,
		},
		'x_lr_discount': 0.5,
		'x_lr_amplifier': 1.0,
		'update_p_at': "all",
		'optimizer_p_fn': "Adam",
		'optimizer_p_kwargs': {
			'lr': 0.00025,
			'weight_decay': 0.01,
		},
		'plot_progress_at': "[]",
	},
	'train_on_batch_kwargs': {
		'is_log_progress': False,
		'is_return_results_every_t': False,
		'is_checking_after_callback_after_t': False,
	},
}

class UModel:
	def __init__(self, config,
                    src_vocab_size,
                    trg_vocab_size,
                    src_pad_idx,
                    trg_pad_idx,
                    embed_dim,
                    n_blocks,
                    n_heads,
                    ff_hid_dim,
                    max_length,
                    dropout,
                    device):

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = Transformer(
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            embed_dim,
            n_blocks,
            n_heads,
            ff_hid_dim,
            max_length,
            dropout,
            self.device
        ).to(self.device)


		self.config['PCTrainer_kwargs']['optimizer_x_fn']=eval(
			'torch.optim.{}'.format(self.config['PCTrainer_kwargs']['optimizer_x_fn'])
		)
		self.config['PCTrainer_kwargs']['optimizer_p_fn']=eval(
			'torch.optim.{}'.format(self.config['PCTrainer_kwargs']['optimizer_p_fn'])
		)
		self.config['PCTrainer_kwargs']['plot_progress_at']=eval(
			self.config['PCTrainer_kwargs']['plot_progress_at']
		)
		self.pc_trainer = pc.PCTrainer(
			self.model,
			**self.config['PCTrainer_kwargs'],
		)

		self.pc_trainer = pc.PCTrainer(
            self.model,
            **self.config['PCTrainer_kwargs'],
        )

		self.model_loss = nn.BCEWithLogitsLoss()
		self.model_optimizer = torch.optim.RMSprop(self.model.parameters(), lr=4e-4)

	def loss_fn(self,outputs, target):
		# print("loss fn called")
		return (outputs - target).pow(2).sum() * 0.5

	def train(self, train_data, test_data):
		for epoch in range(epochs):
			losses = 0
			for (inputs, targets) in train_data:
				inputs = inputs.to(device)
				targets = targets.to(device)
				output = self.model(inputs)
				loss = self.model_loss(output, targets)
				self.model_optimizer.zero_grad()
				loss.backward()
				self.model_optimizer.step()
				losses += loss.item()

			print(f'[{epoch}][Train]', ', losses', losses)
			self.model.eval()
			test_loss = 0
			passed = 0
			k1 = 0
			for (inputs, targets) in test_data:
				k1+=1
				with torch.no_grad():
					inputs = inputs.to(device)
					targets = targets.to(device)
					outputs = self.model(inputs)
					if outputs.argmax() == targets.argmax():
						passed += 1
			self.model.train()
			print("model trained:",k1)
			# print("train data:",train_data)
			k=0
			for (inputs, targets) in train_data:
				inputs = inputs.to(device)
				targets = targets.to(device)
				self.pc_trainer.train_on_batch(
					inputs,self.loss_fn,
					loss_fn_kwargs={
						'target': targets,
					},
					**self.config['train_on_batch_kwargs'],
				)
				k+=1
				print("k:",k)
			print(f'[{epoch}][Test]', ', accuracy', passed / len(dataset_y))

		torch.save(self.model.state_dict(), model_path)

if __name__ == "__main__":
    torch.random.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    n_blocks = 6
    embed_dim = 512
    n_heads = 8
    ff_hid_dim = 4
    max_length = 100
    dropout = 0
    src_pad_idx = 0
    trg_pad_idx = 0
    trg_vocab_size = 20
    src_vocab_size = 20

    src = torch.randint(1, 20, size=(16, 10)).to(device)
    trg = torch.randint(1, 20, size=(16, 10)).to(device)
    print(f'source: {src.cpu().numpy().tolist()}\ntarget: {trg.cpu().numpy().tolist()}')

    # model = Transformer(src_vocab_size,
    #                     trg_vocab_size,
    #                     src_pad_idx,
    #                     trg_pad_idx,
    #                     embed_dim,
    #                     n_blocks,
    #                     n_heads,
    #                     ff_hid_dim,
    #                     max_length,
    #                     dropout,
    #                     device).to(device)

    model = UModel(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_dim,
        n_blocks,
        n_heads,
        ff_hid_dim,
        max_length,
        dropout,
        device
    ).to(device)

    out = model(src, trg)
    print(f'output shape: {out.shape}')
    print(f'output: {out.detach().cpu().numpy().tolist()}')
