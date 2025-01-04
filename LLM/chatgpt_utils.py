import time
import openai

MODEL = "gpt-3.5-turbo"



def chatgpt_response(messages,temperature=0.0):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
            )
            time.sleep(1)
            return response['choices'][0]['message']['content']
        except openai.OpenAIError as e:
            error_msg = str(e)
            if not error_msg.startswith('Rate limit reached'):
                # do not print rate limit error
                print(error_msg)
                print('retrying...')
            time.sleep(30)

