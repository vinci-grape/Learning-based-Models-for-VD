import json

import tiktoken


def openai_tokenize(message, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(message)

results = json.load(open('./result/chatgpt/icl/same_repo/raw_0.0.json'))


cnt = 0
for item in results:
    input = item['input'] + item['output']
    print()
    cnt += len(openai_tokenize( input))


print(cnt)

print(openai_tokenize('hello wadadas dsa dsa das dsa sacsaca s w'))