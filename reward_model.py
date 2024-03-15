import torch
from transformers import pipeline


def get_reward(texts_list, device=""):
    sent_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": 16,
    }
    sentiment_pipe = pipeline(
        "sentiment-analysis", model="lvwerra/distilbert-imdb", device=device
    )

    return sentiment_pipe(texts_list, **sent_kwargs)


if __name__ == "__main__":
    texts_list = [
        "There are laughs and dancing around. I wouldn",
        "About two hundred members of the Missouri National Guard, including two",
        "In the mid-1930s the Trade Unions of England formed with the Lamont Talcott Company,",
        "I shall not waste my time writing more about this subject I wanted to",
        "This show is fairly Mackay-impressive",
        "I cant describe how terrible that movie was, most of the",
        "Years ago, I found a fan of Zwischen and his",
        "I am not sure who is writing this garbage, but if",
        "To bad for a holiday and because (another",
    ]
    print(get_reward(texts_list, device="cpu"))
