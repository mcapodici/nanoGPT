gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters
n_layer = 6
n_head = 6
n_embd = 768
dropout = 0.2
learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small
warmup_iters = 100 # not super necessary potentially
eval_interval = 1000
eval_iters = 20
log_interval = 10
always_save_checkpoint = True
dataset = 'reddit'

out_dir = 'out-reddit-softmax-one'
wandb_log = True
wandb_project = 'nanoGPT_softmax1'
wandb_run_name = 'reddit-mini-gpt-softmax-one'
use_softmax1 = True

compile = False