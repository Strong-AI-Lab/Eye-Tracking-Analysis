import time
import csv

from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_scheduler
from torch.utils.data import DataLoader
from data.mishra import Mishra

FLAGS = flags.FLAGS

flags.DEFINE_integer('num_epochs', None, 'Number of epochs to train the model for.', lower_bound=1, required=True)
flags.DEFINE_integer('batch_size', None, 'The batch size to use.', lower_bound=1, required=True)
flags.DEFINE_float('lr', 5e-5, 'The learning rate to use.', lower_bound=0)
flags.DEFINE_enum('lr_scheduler',  'linear', ['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'], 'Type of learning rate scheduler.')
flags.DEFINE_integer('num_warmup_steps', 0, 'The number of warmup steps to use for learning rate scheduler.', lower_bound=0)
flags.DEFINE_string('tokenizer', 'bert-base-cased', 'The name of the pretrained BERT tokenizer to use. Pretrained models and tokenizers can be found on https://huggingface.co/models.')
flags.DEFINE_string('pretrained_model', None, 'The name of the pretrained BERT model to use. Pretrained models and tokenizers can be found on https://huggingface.co/models.')
flags.DEFINE_boolean('truncation', False, 'Whether to truncate sequence if it is longer than specified max_length. If True, max_length should be set to a reasonable number.')
flags.DEFINE_integer('max_length', 512, 'The maximum length of sequence allowed (the maximum length of the output of the tokenizer).')

flags.DEFINE_integer('num_labels', 2, 'The number of classes.', lower_bound=1)
flags.DEFINE_integer('max_position_embeddings', 512, 'The maximum sequence length that this model might ever be used with.', lower_bound=1)
flags.DEFINE_enum('position_embedding_type',  'absolute', ['absolute', 'relative_key', 'relative_key_query'], 'Type of position embedding.')
flags.DEFINE_integer('hidden_size', 768, 'Dimensionality of the encoder layer and the pooler layer.', lower_bound=1)
flags.DEFINE_integer('num_hidden_layers', 12, 'Number of hidden layers.', lower_bound=1)
flags.DEFINE_integer('num_attention_heads', 12, 'Number of attention heads for each attention layer.', lower_bound=1)
flags.DEFINE_integer('intermediate_size', 3072, 'Dimensionality of the feed-forward layer.', lower_bound=1)
flags.DEFINE_float('hidden_dropout_prob', 0.1, 'The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.', lower_bound=0, upper_bound=1)
flags.DEFINE_float('attention_probs_dropout_prob', 0.0, 'The dropout ratio for the attention probabilities.', lower_bound=0, upper_bound=1)
flags.DEFINE_float('initializer_range', 0.02, 'The standard deviation of the truncated_normal_initializer for initializing all weight matrices.', lower_bound=0)
flags.DEFINE_float('layer_norm_eps', 1e-12, 'The epsilon used by the layer normalization layers.', lower_bound=0)
flags.DEFINE_integer('type_vocab_size', 2, 'The vocabulary size of the token_type_ids passed. Used to tell the model which tokens belong to which sentence or source. Commonly used in classification on pairs of sentences or QA tasks.', lower_bound=0)

flags.DEFINE_boolean('gradient_checkpointing', False, 'If True, use gradient checkpointing to save memory at the expense of slower backward pass.')
flags.DEFINE_boolean('validation', False, 'IMDB dataset does not have valiation set. if set True, a validation set will be created from the training set and use it for evaluation. Otherwise test set will be used for evaluation.')
flags.DEFINE_integer('seed', 0, 'The seed to use for random number generator.')
flags.DEFINE_integer('num_workers', 0, 'The number subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
flags.DEFINE_integer('log_interval', 10, 'The number of training steps in between printing out training loss and accuracy on the terminal.', lower_bound=1)
flags.DEFINE_boolean('use_cpu', False, 'Whether to use CPU.')
flags.DEFINE_string('output', None, 'The output file path.')
flags.DEFINE_string('train_file', None, 'The training data file path.')
flags.DEFINE_string('test_file', None, 'The test data file path.')




def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments or wrong argument names.')

    tokenizer = BertTokenizer.from_pretrained(FLAGS.tokenizer, force_download=False, use_fast=True)



    if FLAGS.pretrained_model:
        # Load pretrained model from huggingface repo
        model = BertForSequenceClassification.from_pretrained(FLAGS.pretrained_model, num_labels=FLAGS.num_labels)
    else:
        # Define our own model and train from scratch.
        # default values can be found here: https://huggingface.co/transformers/model_doc/bert.html#bertconfig
        configs = BertConfig(
            num_labels=FLAGS.num_labels,
            vocab_size=tokenizer.vocab_size,
            hidden_size=FLAGS.hidden_size,
            num_hidden_layers=FLAGS.num_hidden_layers,
            num_attention_heads=FLAGS.num_attention_heads,
            intermediate_size=FLAGS.intermediate_size,
            hidden_act='gelu',
            hidden_dropout_prob=FLAGS.hidden_dropout_prob,
            attention_probs_dropout_prob=FLAGS.attention_probs_dropout_prob,
            max_position_embeddings=FLAGS.max_position_embeddings,
            type_vocab_size=FLAGS.type_vocab_size,
            initializer_range=FLAGS.initializer_range,
            layer_norm_eps=FLAGS.layer_norm_eps,
            pad_token_id=0,
            gradient_checkpointing=FLAGS.gradient_checkpointing,
            position_embedding_type=FLAGS.position_embedding_type,
            is_decoder=False)
        model = BertForSequenceClassification(configs)


    # sum([p.numel() for p in model.parameters()])

    device = torch.device("cuda") if torch.cuda.is_available() and (not FLAGS.use_cpu) else torch.device("cpu")
    model.to(device)


    if FLAGS.validation:
        raise NotImplementedError
    else:
        train_dataset = Mishra(FLAGS.train_file)
        val_dataset = Mishra(FLAGS.test_file)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=FLAGS.batch_size, pin_memory=True, num_workers=FLAGS.num_workers)
    eval_dataloader = DataLoader(val_dataset, batch_size=FLAGS.batch_size, pin_memory=True, num_workers=FLAGS.num_workers)
    
    optimizer = AdamW(model.parameters(), lr=FLAGS.lr)
    num_training_steps = FLAGS.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        FLAGS.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=FLAGS.num_warmup_steps,
        num_training_steps=num_training_steps
)

    for epoch in range(FLAGS.num_epochs):
        start = time.time()
        # train one epoch
        train(epoch, model, tokenizer, optimizer, lr_scheduler, train_dataloader, FLAGS.log_interval, device)
        logging.info('Training one epoch took: %.4f seconds.', time.time()-start)
        
        # evaluate
        eval_acc, eval_loss = eval(model, tokenizer, eval_dataloader, device)
        logging.info('Evaluation after epoch: %d, eval_loss: %.4f, eval_acc: %.4f', epoch,
                   eval_loss, eval_acc)
        
    if FLAGS.output:
        with open(FLAGS.output, 'a') as write_file:
            writer = csv.writer(write_file)
            writer.writerow([epoch, eval_loss, eval_acc])




def train(epoch, model, tokenizer, optimizer, lr_scheduler, dataloader, log_interval, device):
    model.train()
    batch_idx = 0
    with tqdm(total=len(dataloader)) as pbar:
        for texts, labels, _ in dataloader:
            optimizer.zero_grad()
            labels = labels.to(device)
            tokenized = tokenizer(texts, padding="longest", truncation=FLAGS.truncation, return_tensors="pt", max_length=FLAGS.max_length)
            tokenized.to(device)
            outputs = model(labels=labels, **tokenized)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            pbar.update(1)
            if batch_idx % log_interval == 0:
                metric= load_metric("accuracy")
                with torch.no_grad():
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=labels)
                    accuracy = metric.compute()
                    logging.info('Epoch: %d, batch: %d, train_loss: %.4f, train_acc: %.4f', epoch, batch_idx,
                        loss.item(), accuracy['accuracy'])
            batch_idx += 1



def eval(model, tokenizer, dataloader, device):
    metric= load_metric("accuracy")
    loss = 0
    model.eval()
    with tqdm(total=len(dataloader)) as pbar:
        for texts, labels, _ in dataloader:
            # won't truncate long sequences. will throw an error if sequence is longer than the maximum input size of the model.
            tokenized = tokenizer(texts, padding="longest", truncation=FLAGS.truncation, return_tensors="pt", max_length=FLAGS.max_length) 
            tokenized.to(device)
            with torch.no_grad():
                outputs = model(labels=labels.to(device), **tokenized)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=labels)
            loss += outputs.loss.item()
            pbar.update(1)
        
        accuracy = metric.compute()
        loss /= len(dataloader.dataset)
    return accuracy['accuracy'], loss


if __name__ == '__main__':
    app.run(main)