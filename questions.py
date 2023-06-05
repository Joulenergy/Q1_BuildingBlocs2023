import torch
import math
import numpy as np
from model_class import STSBertModel

model = STSBertModel()
model.load_state_dict(torch.load("sts_bert_model.pth"))

test = {
    "GAS-7": {
        "qns": [
            "feeling nervous, anxious, or on edge",
            "unable to stop or control worrying",
            "worrying too much about different things",
            "having trouble relaxing",
            "so restless that it is hard to sit still",
            "becoming easily annoyed or irritable",
            "feeling afraid, as if something awful might happen"
        ],
        "scoring": [
            "I have not been {} at all",
            "I have been {} for several days",
            "I have been {} more than half the days",
            "I have been {} nearly every day"
        ]
    },
    "CUDOS": {
        "qns": [
            "I was bothered by things that usually don't bother me",
            "I did not feel like eating, I wasn't very hungry",
            "I wasn't able to feel happy, even when my family or friends tried to help me feel better",
            "I felt like I was just as good as other kids",
            "I felt like I couldn't pay attention to what I was doing",
            "I felt down and unhappy",
            "I felt like I was too tired to do things",
            "I felt like something good was going to happen",
            "I felt like things I did before didn't work out right",
            "I felt scared",
            "I didn't sleep as well as I usually sleep",
            "I was happy",
            "I was more quiet than usual",
            "I felt lonely, like I didn't have any friends",
            "I felt like kids I know were not friendly or that they didn't want to be with me",
            "I had a good time",
            "I felt like crying",
            "I felt sad",
            "I felt people didn't like me",
            "it was hard to get started doing things"
        ],
        "scoring": [
            "I do not feel that {} at all.",
            "I feel a little bit that {}.",
            "I somewhat feel that {}.",
            "I strongly feel that {}."
        ]
    },
    "EAT-26": {
        "qns": [
            "am terrified about being overweight.",
            "avoid eating when I am hungry.",
            "find myself preoccupied with food.",
            "have gone on eating binges where I feel that I may not be able to stop.",
            "cut my food into small pieces.",
            "aware of the calorie content of foods that I eat.",
            "particularly avoid food with a high carbohydrate content (i.e. bread, rice, potatoes, etc.)",
            "feel that others would prefer if I ate more.",
            "vomit after I have eaten.",
            "feel extremely guilty after eating.",
            "am preoccupied with a desire to be thinner.",
            "think about burning up calories when I exercise.",
            "other people think that I am too thin.",
            "am preoccupied with the thought of having fat on my body",
            "take longer than others to eat my meals.",
            "avoid foods with sugar in them.",
            "eat diet foods.",
            "feel that food controls my life.",
            "display self-control around food.",
            "feel that others pressure me to eat.",
            "give too much time and thought to food.",
            "feel uncomfortable after eating sweets.",
            "engage in dieting behavior.",
            "like my stomach to be empty.",
            "have the impulse to vomit after meals.",
            "enjoy trying new rich foods."
        ],
        "scoring": [
            "I never {}.",
            "I rarely {}.",
            "I sometimes {}.",
            "I often {}.",
            "I usually {}.",
            "I always {}."
        ]
    },
    "IES-R": {
        "qns": [
            "any reminder brought back feelings about it",
            "I had trouble staying asleep",
            "other things kept making me think about it",
            "I felt irritable and angry",
            "I avoided letting myself get upset when I thought about it or was reminded of it",
            "I thought about it when I didn't mean to",
            "I felt as if it hadn't happened or wasn't real",
            "I stayed away from reminders of it",
            "pictures about it popped into my mind",
            "I was jumpy and easily startled",
            "I tried not to think about it",
            "I was aware that I still had a lot of feelings about it, but I didn't deal with them",
            "my feelings about it were kind of numb",
            "I found myself acting or feeling like I was back at that time",
            "I had trouble falling asleep",
            "I had waves of strong feelings about it",
            "I tried to remove it from my memory",
            "I had trouble concentrating",
            "reminders of it caused me to have physical reactions, such as sweating, trouble breathing, nausea, or a pounding heart",
            "I had dreams about it",
            "I felt watchful and on-guard",
            "I tried not to talk about it"
        ],
        "scoring": [
            "I do not feel that {} at all.",
            "I feel that {} a little bit.",
            "I moderately feel that {}.",
            "I feel that {} quite a bit.",
            "I strongly feel that {}."
        ]
    },
    "Y-PSC": {
        "qns": [
            "complain of aches and pains",
            "spend more time alone",
            "tire easily or have little energy",
            "am fidgety or unable to sit still",
            "have trouble with teacher",
            "am less interested in school",
            "act as if driven by a motor",
            "daydream too much",
            "am distracted easily",
            "am afraid of new situations",
            "feel sad or unhappy",
            "am irritable or angry",
            "feel hopeless",
            "have trouble concentrating",
            "am less interested in friends",
            "fight with others",
            "am absent from school",
            "have school grades that are dropping",
            "am down on myself",
            "visit the doctor with the doctor finding nothing wrong",
            "have trouble sleeping",
            "worry a lot",
            "want to be with others more than before",
            "feel that I am bad",
            "take unnecessary risks",
            "get hurt frequently",
            "seem to be having less fun",
            "act younger than children his or her age",
            "disregard rules",
            "suppress my feelings",
            "misunderstand other people's feelings",
            "tease others",
            "blame others for my troubles",
            "take things that do not belong to me",
            "refuse to share",
        ],
        "scoring": [
            "I never {}.",
            "I sometimes {}.",
            "I often {}."
        ]
    },
    "PHQ-9": {
        "qns": [
            "having little interest or pleasure in doing things",
            "feeling down, depressed, or hopeless",
            "having trouble falling or staying asleep, or sleeping too much",
            "feeling tired or having little energy",
            "having poor appetite or overeating",
            "feeling bad about yourself or that you are a failure or have let yourself or your family down",
            "having trouble concentrating on things, such as reading the newspaper or watching television",
            "moving or speaking so slowly that other people could have noticed or being so figety or restless that you have been moving around a lot more than usual",
            "having thoughts that you would be better off dead, or of hurting yourself"
        ],
        "scoring": [
            "I have not been {} at all",
            "I have been {} for several days",
            "I have been {} more than half the days",
            "I have been {} nearly every day"
        ]
    }
}

def score(input_sentence, qns, scoring):
    scores = []
    for qn in qns:
        qn_scores = []
        for i in range(len(scoring)):
            sentence = scoring[i].format(qn)
            qn_scores.append(model.predict_sts([sentence, input_sentence]))
        scores.append(np.array(qn_scores).argmax())
    return scores

def gas_7(input_sentence):
    lst = ["You are suffering from minimal anxiety. That's okay! It's natural for everyone to feel anxious at times, "
           "even those without any mental health problems. It's simply the body's natural response to stress. As long "
           "as you do not suffer from anything major or frequent, these results are still fine, and you do not have "
           "much to worry about.",
           "You are suffering from mild anxiety. Symptoms may include excessive and uncontrollable worry, restlessness, "
           "irritability, difficulty concentrating, muscle tension, sleep disturbances, and a heightened sense of "
           "fear or panic. Fortunately, your anxiety is still quite mild and is not too serious. Some ways to start "
           "combating anxiety include practicing relaxation techniques, engaging in regular exercise, maintaining a "
           "healthy lifestyle and seeking support from loved ones.",
           "You are suffering from moderate anxiety. Symptoms may include excessive worry, restlessness, "
           "irritability, difficulty concentrating, muscle tension, sleep disturbances, and a heightened sense of "
           "fear or panic. Your levels of anxiety are somewhat high and might be dangerous for your health. Luckily, it "
           "is still low enough for you to take matters within your own hands. Some ways to start "
           "combating anxiety include practicing relaxation techniques, engaging in regular exercise, maintaining a "
           "healthy lifestyle and seeking support from loved ones. You might also want to seek support from a therapist.",
           "You are suffering from severe anxiety. Symptoms may include excessive worry, restlessness, "
           "irritability, difficulty concentrating, muscle tension, sleep disturbances, and a heightened sense of "
           "fear or panic. Your levels of anxiety are extremely high, and such high levels of anxiety can be extremely "
           "detrimental to your mental and physical health. Due to such high levels of anxiety, I suggest you seek help "
           "from a professional, such as through challenging negative thoughts through cognitive-behavioral therapy or "
           "medication under the guidance of a healthcare professional.",
           "You are suffering from severe anxiety. Symptoms may include excessive worry, restlessness, "
           "irritability, difficulty concentrating, muscle tension, sleep disturbances, and a heightened sense of "
           "fear or panic. Your levels of anxiety are extremely high, and such high levels of anxiety can be extremely "
           "detrimental to your mental and physical health. Due to such high levels of anxiety, I suggest you seek help "
           "from a professional, such as through challenging negative thoughts through cognitive-behavioral therapy or "
           "medication under the guidance of a healthcare professional."]
    qns = test["GAS-7"]["qns"]
    scoring = test["GAS-7"]["scoring"]
    scores = score(input_sentence, qns, scoring)
    print(scores)
    final_score = math.floor((sum(scores[:-1]) + 3 - scores[-1])/5)
    print(final_score)
    return lst[final_score]

def eat_26(input_sentence):
    qns = test["EAT-26"]["qns"]
    scoring = test["EAT-26"]["scoring"]
    scores = np.array(score(input_sentence, qns, scoring))-2
    scores[scores <= 0] = 0
    print(scores)
    final_score = sum(scores[:-1]) + 3 - scores[-1]
    print(final_score)
    if final_score > 25:
        return "You are most likely suffering from an eating disorder. An eating disorder is when someone begins eating " \
               "too much, or when someone begins to avoid eating, harming their own mental and physical health. These " \
               "usually stem from having low self esteem and a poor body image, and people with eating disorders often " \
               "feel sad and alone. Please get further investigation by a qualified professional."
    return ""

def ies_r(input_sentence):
    qns = test["IES-R"]["qns"]
    scoring = test["IES-R"]["scoring"]
    scores = score(input_sentence, qns, scoring)
    print(scores)
    final_score = sum(scores)
    print(final_score)
    if final_score >= 37:
        return "You are most likely suffering from extremely high levels of PTSD. PTSD is a clinical concern, where " \
               "symptoms include intrusive memories or flashbacks of the traumatic event, avoidance of reminders, " \
               "negative changes in mood and heightened sensitivity of potential threats. Your levels of PTSD high " \
               "enough to suppress your immune system's functioning (even 10 years after an impact event), so please " \
               "seek help from a medical professional as soon as possible."
    if final_score >= 33:
        return "You are most likely suffering from PTSD. PTSD is a clinical concern, where symptoms include intrusive " \
               "memories or flashbacks of the traumatic event, avoidance of reminders, negative changes in mood and " \
               "heightened sensitivity of potential threats. Please seek support from your loved ones or a medical " \
               "professional."
    if final_score >= 24:
        return "PTSD is a clinical concern, where symptoms include intrusive memories or flashbacks of the traumatic " \
               "event, avoidance of reminders, negative changes in mood and heightened sensitivity of potential threats. " \
               "While you might not have full PTSD, you still might have partial PTSD or at least some of the symptoms."
    return ""

def phq_9(input_sentence):
    lst = ["You are suffering from minimal depression. That's okay! It's natural for everyone to feel sad at times, "
           "even those without any mental health problems. As long as you do not suffer from anything major or frequent, "
           "these results are still fine, and you do not have much to worry about.",
           "You are suffering from mild depression. Symptoms may include persistent feelings of sadness, loss of "
           "interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of "
           "worthlessness or guilt, difficulty concentrating, and recurring thoughts of death or suicide. Fortunately, "
           "your depression is still quite mild and is not too serious. Some ways to start combating depression include "
           "engaging in regular exercise, maintaining a healthy lifestyle, seeking support from loved ones, practicing "
           "self-care activities, challenging negative thoughts, and engaging in activities that bring joy and fulfillment.",
           "You are suffering from moderate depression. Symptoms may include persistent feelings of sadness, loss of "
           "interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of "
           "worthlessness or guilt, difficulty concentrating, and recurring thoughts of death or suicide. Your levels of "
           "depression are somewhat high and might be dangerous for your health. Luckily, it is still low enough for you "
           "to take matters within your own hands. Some ways to start combating anxiety include engaging in regular exercise, "
           "maintaining a healthy lifestyle, seeking support from loved ones, practicing self-care activities, "
           "challenging negative thoughts, and engaging in activities that bring joy and fulfillment. You may also want "
           "to seek professional help and take medication, such as antidepressants.",
           "You are suffering from moderately severe depression. Symptoms may include persistent feelings of sadness, loss of "
           "interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of "
           "worthlessness or guilt, difficulty concentrating, and recurring thoughts of death or suicide. Your levels of "
           "depression are somewhat high and might be dangerous for your health. Luckily, it is still low enough for you "
           "to take matters within your own hands. Some ways to start combating anxiety include engaging in regular exercise, "
           "maintaining a healthy lifestyle, seeking support from loved ones, practicing self-care activities, "
           "challenging negative thoughts, and engaging in activities that bring joy and fulfillment. You may also want "
           "to seek professional help and take medication, such as antidepressants.",
           "You are suffering from severe depression. Symptoms may include persistent feelings of sadness, loss of "
           "interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of "
           "worthlessness or guilt, difficulty concentrating, and recurring thoughts of death or suicide. Your levels of "
           "depression are extremely high, and such high levels of depression can be extremely detrimental to your "
           "mental and physical health. Due to such high levels of depression, I suggest you seek help from a professional, "
           "such as through challenging negative thoughts through cognitive-behavioral therapy or medication under the "
           "guidance of a healthcare professional.",
           "You are suffering from severe depression. Symptoms may include persistent feelings of sadness, loss of "
           "interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of "
           "worthlessness or guilt, difficulty concentrating, and recurring thoughts of death or suicide. Your levels of "
           "depression are extremely high, and such high levels of depression can be extremely detrimental to your "
           "mental and physical health. Due to such high levels of depression, I suggest you seek help from a professional, "
           "such as through challenging negative thoughts through cognitive-behavioral therapy or medication under the "
           "guidance of a healthcare professional."]
    qns = test["PHQ-9"]["qns"]
    scoring = test["PHQ-9"]["scoring"]
    scores = score(input_sentence, qns, scoring)
    print(scores)
    final_score = math.floor(sum(scores)/5)
    print(final_score)
    return lst[final_score]

def y_psc(input_sentence):
    qns = test["Y-PSC"]["qns"]
    scoring = test["Y-PSC"]["scoring"]
    scores = score(input_sentence, qns, scoring)
    print(scores)
    final_score = sum(scores)
    print(final_score)
    if final_score>=30:
        return "psychological impairmment"

def cudos(input_sentence):
    qns = test["CUDOS"]["qns"]
    scoring = test["CUDOS"]["scoring"]
    scores = score(input_sentence, qns, scoring)
    print(scores)
    final_score = sum(scores[0:3]) + sum(scores[4:7]) + sum(scores[8:11]) + sum(scores[12:15]) + 12 - scores[3] - scores[7] - scores[11] - scores[15]
    print(final_score)
    if final_score>15:
        return "depressive symptoms"
    return None
