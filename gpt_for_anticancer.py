from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import selfies as sf
import selfies
import torch
#from torchsummary import summary

## check environment
print(torch.cuda.is_available())
print(torch.Tensor([1,2]).cuda())


tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")

#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
##model = AutoModelForCausalLM.from_pretrained("./ChemGPT47/")
model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-4.7M")


#tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
#model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-4.7M")
print(tokenizer)
print(model)

## load anti-cancer smiles

train_raw = open("datasets/train.raw").readlines()
valid_raw = open("datasets/validation.raw").readlines()

train_encode =[]
for i in train_raw:
    #if sf.encoder(i.strip()) is None:
    #    print (i)
    try:
        if sf.encoder(i.strip()) is not None:
           train_encode.append(sf.encoder(i.strip()))
    except:
        print("error smiles ",i.strip())

valid_encode =[]
for i in valid_raw:
    try: 
        if sf.encoder(i.strip()) is not None:
           valid_encode.append(sf.encoder(i.strip()))
    except:
        print("error smiles ",i.strip())
'''       
f=open("datasets/train_datasets.txt","w")
#print (train_encode)
f.write("\n".join(train_encode))
f.close()

f=open("datasets/valid_datasets.txt","w")
f.write("\n".join(valid_encode))
f.close()
'''
#print(train_encode)

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="datasets/train_datasets.txt",  # Replace with the path to your training dataset
    block_size=128  # Adjust the block size according to your needs
)

valid_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="datasets/valid_datasets.txt",  # Replace with the path to your training dataset
    block_size=128  # Adjust the block size according to your needs
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're not doing masked language modeling (MLM) for text generation
)

training_args = TrainingArguments(
    output_dir="./log/text-generation-model",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=64,
    save_steps=100,
    save_total_limit=2,
)



trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset
)

trainer.train()

# You can evaluate the fine-tuned model and save it if needed
results = trainer.evaluate()
print(results)
trainer.save_model("./output_model")
#tokenizer = tokenizer.train_new_from_iterator(get_training_corpus(), 52000)
#tokenizer.save_pretrained("your-tokenizer")










