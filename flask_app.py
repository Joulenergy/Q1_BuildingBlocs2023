from flask import Flask, request, render_template

from questions import gas_7, eat_26, ies_r, phq_9
from chatgpt import respond

prompt_start = "I want you to act as a mental health chatbot. The user will talk about their life and the problems " \
               "that they are facing, including but not limited to their mental health, stress, insecurities, " \
               "thoughts and actions. Console them and give them advice. Share possible solutions that might be able " \
               "to help them improve their mental health. The user will start by talking about their situation: "

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
    return render_template('chat.html')

history = []
text = ""
@app.route('/get')
def get():
    print(len(history))
    prompt = request.args.get('msg')
    if len(history) != 20:
        bot_text = respond(prompt, history)
    else:
        history.append({"role": "user", "content": prompt})
        text = ""
        for i in history[::2]:
            text += i["content"] + " "
        print(text)
        bot_text = "I see, thanks for sharing. Here is my diagnosis:\n\n" + gas_7(text) + "\n\n" + phq_9(text)
        eat = eat_26(text)
        if eat != "":
            bot_text += "\n\n" + eat
        ies= ies_r(text)
        if ies != "":
            bot_text += "\n\n" + ies
        bot_text += "\n\nThank you for chatting with me! If you would like to, we could end the conversation here, or " \
                    "you could continue chatting with me. I can answer whatever questions you may have or give more " \
                    "advice if you want."
        history.append({"role": "assistant", "content": bot_text})
        start = history[0]["content"][545:]
        history[0]["content"] = prompt_start + start
    return {'bot': bot_text}

app.run(debug=True)