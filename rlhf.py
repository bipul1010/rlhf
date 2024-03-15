import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from reward_model import get_reward
import wandb

# import wandb
from tqdm import tqdm

wandb.init()


def build_dataset(
    config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8
):

    print(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})

    ds = ds.filter(lambda x: len(x["review"]) > 200)
    input_size = LengthSampler(
        min_value=input_min_text_length, max_value=input_max_text_length
    )

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device=device)
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )
    trained_model_path = ""
    tokenizer_path = ""

    dataset = build_dataset(config)

    # ##model going to fine-tune
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    # ## this is the reference  - frozen model
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # input_ids = tokenizer(
    #     ["how are  you?", "all good?"], return_tensors="pt", padding=True
    # )
    # print(input_ids)
    # generate_ids = model.generate(input_ids=input_ids["input_ids"], max_length=50)
    # print(generate_ids)
    # print(tokenizer.batch_decode(generate_ids))

    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    ppo_trainer = PPOTrainer(
        config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator
    )

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]  ##(B,Seq) - >List

        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs[
                "max_new_tokens"
            ] = gen_len  # Number of tokens to generate (chosen randomly)
            response = ppo_trainer.generate(query, **response_generation_kwargs)
            response = response.squeeze(0)
            response_tensors.append(response[-gen_len:])
        batch["response"] = [tokenizer.decode(r) for r in response_tensors]

        rewards_texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        rewards_output = get_reward(rewards_texts, device)
        rewards = [
            torch.tensor(k[1]["score"]) for k in rewards_output
        ]  ##get scores of only positive reviews

        ##Phase 1 + Phase 2: compute the logprobability and run the PPO update
        stats = ppo_trainer.step(
            query_tensors, response_tensors, rewards
        )  ##go to ppo_step.txt for pseudo code.

        ppo_trainer.log_stats(stats, batch, rewards)

    model.save_pretrained(f"{trained_model_path}/gpt2-imdb-pos-v2", push_to_hub=False)
    tokenizer.save_pretrained(f"{tokenizer_path}/gpt2-imdb-pos-v2", push_to_hub=False)
