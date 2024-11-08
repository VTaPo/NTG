import argparse

from crawdata import get_topic
from generate_text import text_generator
from utils import save_to_msword

def main(args):
    topic = args.topic
    output_len = args.output_len
    title, raw_text = get_topic(topic)
    generated_text = text_generator(raw_text, output_len)
    article = title + generated_text
    save_to_msword(article)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text using OpenAI API')
    parser.add_argument('--topic', type=str, help='The topic to generate text for', required=True)
    parser.add_argument('--output_len', type=int, help='The length of the output text', required=True)
    args = parser.parse_args()
    main(args)