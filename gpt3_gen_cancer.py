# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

import selfies
tokenizer = AutoTokenizer.from_pretrained("ncfrey/ChemGPT-4.7M")
model = AutoModelForCausalLM.from_pretrained("output_model").to('cuda')
import torch
# 准备输入文本，您应该根据模型的预期输入来修改它
input_text = "C"  # 假设我们开始一个化学式

# 编码输入文本为 token id
input_ids = tokenizer.encode(input_text, return_tensors="pt")

batch_size = 1000  # 每批生成的序列数量
total_sequences = 200000  # 需要生成的总序列数量
num_files = 50  # 存储序列的文件数量


fw = None
file_counter = 1
sequence_counter = 1


for i in range(200):
    torch.cuda.empty_cache()
    if i % 20 == 0:
        # 关闭前一个文件（如果有的话）
        if fw:
            fw.close()
        
        # 打开新文件
        filename = f"generated_smiles_{file_counter}.txt"
        fw = open(filename, "w")
        file_counter += 1
    
    output_sequences = model.generate(
        input_ids=input_ids.to('cuda'),
        max_length=100,
        temperature=1.0,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        num_return_sequences=batch_size,
    )
    generated_sequences = [
    tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True)
    for output_sequence in output_sequences
    ]

    for generated_selfies in generated_sequences:
       filtered_selfies = generated_selfies.replace("[CLS]", "").replace("[SEP]", "").replace("[UNK]", "").replace(' ','')
       #sequence_counter = 1


       try:
          generated_smiles = selfies.decoder(filtered_selfies)
          #print(f"Generated SMILES {sequence_counter}: {generated_smiles}")
          smiles_name = f"SMILES_{sequence_counter}"
          sequence_counter += 1
          fw.write(f"{smiles_name}: {generated_smiles}\n")
       except selfies.exceptions.DecoderError as e:
          print("An error occurred during decoding:", e)

fw.close()
