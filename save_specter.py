from transformers import AutoTokenizer, AutoModel
from memory_profiler import profile

@profile
def main():
    model = AutoModel.from_pretrained("allenai/specter")
    model.save_pretrained("./model", max_shard_size="200MB")
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    tokenizer.save_pretrained("./model")
    new_model = AutoModel.from_pretrained("./model", low_cpu_mem_usage=True)

if __name__ == "__main__":
    main()