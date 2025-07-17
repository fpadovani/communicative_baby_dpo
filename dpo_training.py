from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import os

# Load model and tokenizer
model_name = "bbunzeck/another-llama"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
dataset = load_dataset("fpadovani/child-dpo-preferences-synthetic", split="train")

# Define DPO training config
training_args = DPOConfig(
    output_dir="./dpo_outputs_complete_synthetic",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=10,
    logging_steps=10,
    save_steps=2000,
    logging_dir="./dpo_outputs_complete_synthetic/logs",
    report_to=[]
)


class SaveEveryXTokensCallback(TrainerCallback):
    def __init__(self, tokens_per_save=10_000_000, max_tokens=100_000_000, output_dir="./checkpoints"):
        self.tokens_per_save = tokens_per_save
        self.max_tokens = max_tokens
        self.output_dir = output_dir
        self.total_tokens = 0
        self.next_save_threshold = tokens_per_save

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        batch = kwargs.get("inputs", {})
        
        # 🔍 Inspect keys present in the batch
        if state.global_step == 1:  # or just always print if you prefer
            print(f"Batch keys at step {state.global_step}: {list(batch.keys())}")
        
        chosen_input_ids = batch.get("chosen_input_ids")
        rejected_input_ids = batch.get("rejected_input_ids")
        
        batch_tokens = 0
        if chosen_input_ids is not None:
            batch_tokens += chosen_input_ids.numel()
        if rejected_input_ids is not None:
            batch_tokens += rejected_input_ids.numel()
        
        self.total_tokens += batch_tokens

        print(f"Tokens this step: chosen={chosen_input_ids.numel() if chosen_input_ids is not None else 0}, "
            f"rejected={rejected_input_ids.numel() if rejected_input_ids is not None else 0}, "
            f"total={self.total_tokens}")

        

class FullMetricsLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.keys_written = False
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return

        # Initialize and write header if first time
        if not self.keys_written:
            self.keys = ["step"] + list(logs.keys())
            with open(self.log_file_path, "w") as f:
                f.write(",".join(self.keys) + "\n")
            self.keys_written = True

        # Write metric row
        row = [str(state.global_step)] + [str(logs.get(k, "")) for k in self.keys[1:]]
        with open(self.log_file_path, "a") as f:
            f.write(",".join(row) + "\n")


# Initialize DPO trainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer
)

# Add logging callback
trainer.add_callback(FullMetricsLoggerCallback("./dpo_outputs_complete_synthetic/logs/training_metrics.csv"))
# Train!
trainer.train()
