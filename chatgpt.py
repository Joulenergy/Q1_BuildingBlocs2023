import requests

url = "https://api.openai.com/v1/chat/completions"
api_key = "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"

prompt_start = "I want you to act as a listener. The user will talk about their life and the problems that they are " \
               "facing, including but not limited to their mental health, stress, insecurities, thoughts and actions. " \
               "End every response with one question about the situation and problems that they are in, such that you " \
               "get as much information as possible. Ask one question at a time. However, do not share possible " \
               "solutions that might be able to help them improve their mental health, only ask questions. The user " \
               "will start by talking about their situation: "

headers = {
    'Accept': 'text/event-stream',
    'Authorization': 'Bearer ' + api_key
}

def respond(prompt, history):
    if len(history) == 0:
        history.append({"role": "system", "content": prompt_start + prompt})
    else:
        history.append({"role": "user", "content": prompt})
    data = {
        "model": "gpt-3.5-turbo",
        "messages": history,
        "temperature": 1
    }
    response = requests.post(url, stream=True, headers=headers, json=data).json()["choices"][0]["message"]["content"]
    history.append({"role": "assistant", "content": response})
    return response
