```python
import sys
sys.path.append("..")
import torch
import random
from transformers import *
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from pytorch_constraints.constraint import constraint
from pytorch_constraints.sampling_solver import WeightedSamplingSolver
#from pytorch_constraints.circuit_solver import SemanticLossCircuitSolver
from pytorch_constraints.shaped_lazy_solver import ProductTNormSolver
from pytorch_constraints.shaped_lazy_solver import GodelTNormSolver
from pytorch_constraints.shaped_lazy_solver import LukasiewiczTNormSolver

```


```python
LABEL_TO_ID = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
ENT = LABEL_TO_ID['entailment']
CON = LABEL_TO_ID['contradiction']
NEU = LABEL_TO_ID['neutral']

def process_data(tokenizer, path):
    def batch_encode(all_p, all_h):
        return tokenizer.batch_encode_plus([(p, h) for p, h in zip(all_p, all_h)], max_length=100, padding=True, return_tensors='pt')

    files = ['snli.train.p.txt', 'snli.train.h.txt', 'snli.train.label.txt', \
        'mscoco.train.p.txt', 'mscoco.train.h.txt', 'mscoco.train.z.txt', \
        'mscoco.test.p.txt', 'mscoco.test.h.txt', 'mscoco.test.z.txt', \
        'snli.val.p.txt', 'snli.val.h.txt', 'snli.val.label.txt']
    all_data = []
    for file in files:
        all_data.append([])
        file = path + '/' + file
        with open(file, 'r') as f:
            print('loading from', file)
            for line in f:
                if line.strip() == '':
                    continue
                all_data[-1].append(line.strip())

    snli_train = batch_encode(all_data[0], all_data[1])['input_ids']
    snli_train_label = torch.tensor([LABEL_TO_ID[l] for l in all_data[2]], dtype=torch.long)
    mscoco_train_ph = batch_encode(all_data[3], all_data[4])['input_ids']
    mscoco_train_hz = batch_encode(all_data[4], all_data[5])['input_ids']
    mscoco_train_pz = batch_encode(all_data[3], all_data[5])['input_ids']
    mscoco_test_ph = batch_encode(all_data[6], all_data[7])['input_ids']
    mscoco_test_hz = batch_encode(all_data[7], all_data[8])['input_ids']
    mscoco_test_pz = batch_encode(all_data[6], all_data[8])['input_ids']
    snli_test = batch_encode(all_data[9], all_data[10])['input_ids']
    snli_test_label = torch.tensor([LABEL_TO_ID[l] for l in all_data[11]], dtype=torch.long)

    snli_train = TensorDataset(snli_train, snli_train_label)
    mscoco_train = TensorDataset(mscoco_train_ph, mscoco_train_hz, mscoco_train_pz)
    mscoco_test = TensorDataset(mscoco_test_ph, mscoco_test_hz, mscoco_test_pz)
    snli_test = TensorDataset(snli_test, snli_test_label)
    return snli_train, mscoco_train, mscoco_test, snli_test
```


```python
def evaluate(model, trans_data, test_data, batch_size=8, device=torch.device('cpu')):
    trans_data_loader = DataLoader(trans_data, sampler=SequentialSampler(trans_data), batch_size=batch_size)
    ex_cnt = 0
    global_sat = 0.0
    conditional_sat = []
    model = model.to(device)
    for _, batch in enumerate(trans_data_loader):
        with torch.no_grad():
            logits_ph = model(input_ids=batch[0].to(device), return_dict=True).logits
            logits_hz = model(input_ids=batch[1].to(device), return_dict=True).logits
            logits_pz = model(input_ids=batch[2].to(device), return_dict=True).logits
    
            ph_y = torch.softmax(logits_ph.view(-1, len(LABEL_TO_ID)), dim=-1)
            hz_y = torch.softmax(logits_hz.view(-1, len(LABEL_TO_ID)), dim=-1)
            pz_y = torch.softmax(logits_pz.view(-1, len(LABEL_TO_ID)), dim=-1)
    
            ph_y_mask = (ph_y == ph_y.max(-1)[0].unsqueeze(-1))
            hz_y_mask = (hz_y == hz_y.max(-1)[0].unsqueeze(-1))
            pz_y_mask = (pz_y == pz_y.max(-1)[0].unsqueeze(-1))
    
            lhs, satisfied = transitivity_check(ph_y_mask, hz_y_mask, pz_y_mask)
            global_sat += float(satisfied.sum())
            conditional_sat.extend([float(s) for l, s in zip(lhs, satisfied) if l])
            ex_cnt += batch[0].shape[0]
        
    print('Global percent of predictions that violate the transitivity constraint', 
        1-global_sat/ex_cnt)
    conditional_sat = sum(conditional_sat)/len(conditional_sat) if len(conditional_sat) != 0 else 1
    print('Conditional percent of predictions that violate the transitivity constraint', 
        1-conditional_sat)

    test_data_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)
    ex_cnt = 0
    correct_cnt = 0
    for _, batch in enumerate(test_data_loader):
        with torch.no_grad():
            logits_ph = model(input_ids=batch[0].to(device), return_dict=True).logits
            pred_ph = logits_ph.cpu().argmax(-1)
            gold_ph = batch[1]
            correct_cnt += int((pred_ph == gold_ph).sum().item())
            ex_cnt += batch[0].shape[0]
    print('test set accuracy', correct_cnt/ex_cnt)
```


```python

# the actual constraint function used for t-norm solvers
# inputs are logits predictions
def transitivity(ph_batch, hz_batch, pz_batch):
    ee_e = (ph_batch[:, ENT]).logical_and(hz_batch[:, ENT]) <= (pz_batch[:, ENT])
    ec_c = (ph_batch[:, ENT]).logical_and(hz_batch[:, CON]) <= (pz_batch[:, CON])
    ne_notc = (ph_batch[:, NEU]).logical_and(hz_batch[:, ENT]) <= (pz_batch[:, CON]).logical_not()
    nc_note = (ph_batch[:, CON]).logical_and(hz_batch[:, NEU]) <= (pz_batch[:, CON]).logical_not()
    return ee_e.logical_and(ec_c).logical_and(ne_notc).logical_and(nc_note)

# checking if constraint is satisfied
# inputs are binary tensors
def transitivity_check(ph_y_mask, hz_y_mask, pz_y_mask):
    ee = ph_y_mask[:, ENT].logical_and(hz_y_mask[:, ENT])
    ec = ph_y_mask[:, ENT].logical_and(hz_y_mask[:, CON])
    ne = ph_y_mask[:, NEU].logical_and(hz_y_mask[:, ENT])
    nc = ph_y_mask[:, NEU].logical_and(hz_y_mask[:, CON])

    ee_e = ee.logical_not().logical_or(pz_y_mask[:, ENT])
    ec_c = ec.logical_not().logical_or(pz_y_mask[:, CON])
    ne_notc = ne.logical_not().logical_or(pz_y_mask[:, CON].logical_not())
    nc_note = nc.logical_not().logical_or(pz_y_mask[:, ENT].logical_not())

    lhs = ee.logical_or(ec).logical_or(ne).logical_or(nc)
    return lhs, ee_e.logical_and(ec_c).logical_and(ne_notc).logical_and(nc_note)
```


```python
# model: transformer for sequence classification
def train(model, solver, constraint_func, train_gold, train_trans, 
            lr=5e-5, batch_size=8, seed=1, grad_clip=1.0, lambda_trans=1, epoch=1,
            use_gold=True, use_trans=True,
            device=torch.device('cpu')):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    train_gold_loader = DataLoader(train_gold, sampler=RandomSampler(train_gold), batch_size=batch_size)
    train_trans_loader = DataLoader(train_trans, sampler=RandomSampler(train_trans), batch_size=batch_size)

    # mixing two datasets
    data_loaders = [train_gold_loader, train_trans_loader]
    expanded_data_loader = [train_gold_loader] * len(train_gold_loader) + [train_trans_loader] * len(train_trans_loader)
    random.shuffle(expanded_data_loader)

    # create optimizer
    weight_decay = 0
    no_decay = ['bias', 'LayerNorm.weight']
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    total_updates = epoch * len(expanded_data_loader)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_updates)

    update_cnt = 0
    loss_accumulator = 0.0
    model = model.to(device)
    model.zero_grad()
    for epoch_id in range(epoch):
        iters = [loader.__iter__() for loader in data_loaders]
        for loader in expanded_data_loader:
            batch = next(iters[data_loaders.index(loader)])

            # if current batch is labeled data
            if loader is train_gold_loader:
                if not use_gold:
                    continue
                output = model(input_ids=batch[0].to(device), labels=batch[1].to(device), return_dict=True)
                loss = output[0]

            elif loader is train_trans_loader:
                if not use_trans:
                    continue
                logits_ph = model(input_ids=batch[0].to(device), return_dict=True).logits
                logits_hz = model(input_ids=batch[1].to(device), return_dict=True).logits
                logits_pz = model(input_ids=batch[2].to(device), return_dict=True).logits

                constrain_func = constraint(constraint_func, solver)
                loss = constrain_func(logits_ph, logits_hz, logits_pz)
                loss = loss * lambda_trans

            else:
                raise Exception('unrecognized loader')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            model.zero_grad()

            loss_accumulator += (loss.item())
            update_cnt += 1

            if update_cnt % 100 == 0:
                print('trained {0} steps, avg loss {1:4f}'.format(update_cnt, float(loss_accumulator/update_cnt)))

    return model
```


```python
#device = torch.device('cpu')
device = torch.device("cuda", 0)

print('initializing models and solvers...')
constraint_func = transitivity
solver = ProductTNormSolver()

config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = len(LABEL_TO_ID)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
```


```python
# preprocess and batch up data
print('processing data...')
snli_train, mscoco_train, mscoco_test, snli_test = process_data(tokenizer, './nli/')
```


```python
# Now let us train a distilbert NLI model and evaluate it on the transitivity test set
# The pipeline is the following:
# 1. first train a model purely on labeled SNLI data, this will give us a model with reasonably good accuracy
# 2. continue training this model on the union of labeled SNLI data and transitivity training split,
#      the training loss will be cross entropy and constraint loss
#
# The expectation is to see whether adding constraint will help reducint transitivity violations 
#    i.e. comparing step 1 and step 2, we should see violation drop without trading off accuracy
# We will also train a baseline model (w/o constraint and transitivity data) for comparison.
#

print('training baseline model...')
print('step 1')
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=5e-5, epoch=1, use_gold=True, use_trans=False, device=device)
evaluate(model, mscoco_test, snli_test, device=device)
print('step 2')
# note that we set use_trans=False, so the constraint will not be used
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=2e-5, epoch=1, use_gold=True, use_trans=False, device=device)
evaluate(model, mscoco_test, snli_test, device=device)

# training baseline model...
# step 1
# trained 100 steps, avg loss 1.099848
# trained 200 steps, avg loss 1.030215
# trained 300 steps, avg loss 0.965319
# trained 400 steps, avg loss 0.911978
# trained 500 steps, avg loss 0.875783
# trained 600 steps, avg loss 0.858009
# Global percent of predictions that violate the transitivity constraint 0.08020000000000005
# Conditional percent of predictions that violate the transitivity constraint 0.21216931216931212
# test set accuracy 0.7552
# step 2
# trained 100 steps, avg loss 0.680515
# trained 200 steps, avg loss 0.588770
# trained 300 steps, avg loss 0.522388
# trained 400 steps, avg loss 0.494158
# trained 500 steps, avg loss 0.475138
# trained 600 steps, avg loss 0.466655
# Global percent of predictions that violate the transitivity constraint 0.0726
# Conditional percent of predictions that violate the transitivity constraint 0.19568733153638818
# test set accuracy 0.757
```


```python
print('training model with product t-norm solver...')
print('step 1')
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=5e-5, epoch=1, use_gold=True, use_trans=False, device=device)
evaluate(model, mscoco_test, snli_test, device=device)
print('step 2')
# Now we set use_trans=True, so the constraint will be used
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=2e-5, epoch=1, use_gold=True, use_trans=True, device=device)
evaluate(model, mscoco_test, snli_test, device=device)

# For reference, the log should look like the following:
# training model with product t-norm solver...
# step 1
# trained 100 steps, avg loss 1.108301
# trained 200 steps, avg loss 1.078871
# trained 300 steps, avg loss 1.043484
# trained 400 steps, avg loss 1.001291
# trained 500 steps, avg loss 0.960916
# trained 600 steps, avg loss 0.939840
# Global percent of predictions that violate the transitivity constraint 0.12919999999999998
# Conditional percent of predictions that violate the transitivity constraint 0.26185650587758413
# test set accuracy 0.7202
# step 2
# trained 100 steps, avg loss 0.461444
# trained 200 steps, avg loss 0.414778
# trained 300 steps, avg loss 0.386260
# trained 400 steps, avg loss 0.368699
# trained 500 steps, avg loss 0.356729
# trained 600 steps, avg loss 0.341195
# trained 700 steps, avg loss 0.339344
# trained 800 steps, avg loss 0.326640
# trained 900 steps, avg loss 0.321558
# trained 1000 steps, avg loss 0.315082
# trained 1100 steps, avg loss 0.311681
# trained 1200 steps, avg loss 0.308697
# trained 1300 steps, avg loss 0.300718
# Global percent of predictions that violate the transitivity constraint 0.02400000000000002
# Conditional percent of predictions that violate the transitivity constraint 0.11100832562442187
# test set accuracy 0.7498
```


```python
# Now let's try sampling solver
#
# the constraint function specifically for sampling solvers
def transitivity_sampling(ph_batch, hz_batch, pz_batch):
    # TODO, can't do this:
    # ENT = LABEL_TO_ID['Entailment']
    ee_e = (ph_batch == 0).logical_and(hz_batch == 0) <= (pz_batch == 0)
    ec_c = (ph_batch == 0).logical_and(hz_batch == 1) <= (pz_batch == 1)
    ne_notc = (ph_batch == 2).logical_and(hz_batch == 0) <= (pz_batch == 1).logical_not()
    nc_note = (ph_batch == 2).logical_and(hz_batch == 1) <= (pz_batch == 0).logical_not()
    # just block Neu and Neu -> Neu and force it to change
    block_safezone = (ph_batch == 2).logical_and(hz_batch == 2) <= (pz_batch == 2).logical_not()
    return ee_e.logical_and(ec_c).logical_and(ne_notc).logical_and(nc_note).logical_and(block_safezone)


print('initializing models and solvers...')
constraint_func = transitivity_sampling
solver = WeightedSamplingSolver(50)

config = AutoConfig.from_pretrained('distilbert-base-uncased')
config.num_labels = len(LABEL_TO_ID)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
```


```python
print('training model with sampling solver...')
print('step 1')
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=5e-5, epoch=1, use_gold=True, use_trans=False, device=device)
evaluate(model, mscoco_test, snli_test, device=device)
print('step 2')
model = train(model, solver, constraint_func, snli_train, mscoco_train, lr=2e-5, epoch=1, use_gold=True, use_trans=True, device=device)
evaluate(model, mscoco_test, snli_test, device=device)

# for reference, the log should look like the following:
# training model with sampling solver...
# step 1
# trained 100 steps, avg loss 1.095759
# trained 200 steps, avg loss 1.013510
# trained 300 steps, avg loss 0.944434
# trained 400 steps, avg loss 0.900531
# trained 500 steps, avg loss 0.862019
# trained 600 steps, avg loss 0.844901
# Global percent of predictions that violate the transitivity constraint 0.06499999999999995
# Conditional percent of predictions that violate the transitivity constraint 0.176056338028169
# test set accuracy 0.7464
# step 2
# training on transitivity data...
# trained 100 steps, avg loss 0.652288
# trained 200 steps, avg loss 0.508393
# trained 300 steps, avg loss 0.423571
# trained 400 steps, avg loss 0.377770
# trained 500 steps, avg loss 0.344454
# trained 600 steps, avg loss 0.320258
# trained 700 steps, avg loss 0.307840
# trained 800 steps, avg loss 0.295266
# trained 900 steps, avg loss 0.283681
# trained 1000 steps, avg loss 0.278062
# trained 1100 steps, avg loss 0.274637
# trained 1200 steps, avg loss 0.271620
# trained 1300 steps, avg loss 0.261367
# Global percent of predictions that violate the transitivity constraint 0.010000000000000009
# Conditional percent of predictions that violate the transitivity constraint 0.08561643835616439
# test set accuracy 0.7576
```
