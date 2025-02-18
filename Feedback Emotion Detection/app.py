import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

##d:\Text-Emotion-Detection-main\Text Emotion Detection\

pipe_lr = joblib.load(open(r"model\text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

emotional_keywords = {
    'anger', 'disgust', 'fear', 'happy', 'joy', 'neutral', 'sad', 'sadness', 'shame', 'surprise', 
    'love', 'hate', 'excited', 'bored', 'anxiety', 'relaxed', 'enthusiastic', 'confident', 'overwhelmed', 
    'content', 'frustrated', 'optimistic', 'pessimistic', 'hopeful', 'guilty',
    'anxious', 'devastated', 'elated', 'envious', 'grateful', 'hopeful', 'indifferent', 'irritated',
    'lonely', 'motivated', 'nostalgic', 'proud', 'regretful', 'rejuvenated', 'satisfied', 'skeptical',
    'stressed', 'triumphant', 'uneasy', 'victorious', 'vulnerable', 'weary', 'zealous', 'dismayed', 
    'inspired', 'desperate', 'ashamed', 'intrigued', 'conflicted', 'tired', 'doubtful', 'scared', 
    'calm', 'cheerful', 'playful', 'hope', 'resentful', 'touched', 'worried', 'suspicious', 
    'overjoyed', 'detached', 'empathetic', 'cautious', 'betrayed', 'sympathetic', 'disappointed', 
    'irate', 'intense', 'enthused', 'passionate', 'hopeful', 'impatient', 'nostalgic', 'apprehensive',
    'exhausted', 'relieved', 'unmotivated', 'curious', 'indignant', 'shocked', 'calm', 'blissful', 
    'heartbroken', 'serene', 'eager', 'troubled', 'enlightened', 'sorrowful', 'suspenseful', 'humiliated', 
    'satisfied', 'disheartened', 'scornful', 'jubilant', 'alienated', 'delighted', 'gloomy', 'startled', 
    'astonished', 'bewildered', 'sensitive', 'cheated', 'disillusioned', 'exhilarated', 'resolved', 
    'tension', 'distressed', 'restless', 'compassionate', 'calm', 'adventurous', 'motivated', 
    'unsettled', 'reassured', 'uncomfortable', 'encouraged', 'worried', 'infuriated', 'discontent',
    'fascinated', 'unsure', 'cheerful', 'intimidated', 'compelled', 'inspired', 'disheartened',
    'frustrated', 'overwhelmed', 'apathetic', 'irritated', 'satisfied', 'suspicious', 'unconfident',
    'nervous', 'shocked', 'resentful', 'fearful', 'disgusted', 'torn', 'resolved', 'nostalgic',
    'terrified', 'discontented', 'happy', 'unhappy', 'wistful', 'forlorn', 'courageous', 'contented',
    'bewildered', 'stimulated', 'jaded', 'fulfilled', 'satisfied', 'emotional', 'infatuated',
    'hopeless', 'bitter', 'challenged', 'overcome', 'challenged', 'cautious', 'miserable', 
    'sadistic', 'cheerful', 'joyful', 'longing', 'euphoric', 'hostile', 'relieved', 'worried',
    'surreal', 'insulted', 'betrayed', 'humble', 'insecure', 'pessimistic', 'bewildered', 
    'melancholic', 'apologetic', 'overjoyed', 'disillusioned', 'hopeful', 'impressed', 'exhausted',
    'confounded', 'incredulous', 'overstimulated', 'encumbered', 'unmotivated', 'jubilant',
    'apprehensive', 'feisty', 'flustered', 'apprehensive', 'exasperated', 'bewildered', 'infuriated',
    'vulnerable', 'nervous', 'tranquil', 'reluctant', 'silly', 'joyful', 'tired', 'energetic',
    'provoked', 'crushed', 'unfazed', 'baffled', 'lighthearted', 'introspective', 'abandoned', 
    'optimistic', 'disparaged', 'disgraced', 'perplexed', 'tense', 'indifferent', 'indignant',
    'contemplative', 'startled', 'apprehensive', 'apprehensive'
}


def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def plot_custom_wordcloud(texts):
    all_texts = " ".join(texts)

    if not all_texts.strip(): 
        st.warning("No text available to generate the WordCloud.")
        return  

    words = re.findall(r'\w+', all_texts.lower())  # Get words and convert to lowercase
 
    filtered_words = [word for word in words if word in emotional_keywords and word not in ENGLISH_STOP_WORDS]


    if filtered_words:
       
        word_freq = Counter(filtered_words)

        def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            return "hsl({}, 100%, 30%)".format(np.random.randint(0, 360))

        # Generate the WordCloud
        wordcloud = WordCloud(width=800, height=400, 
                              background_color="black",  
                              color_func=custom_color_func,  
                              max_font_size=120,  
                              min_font_size=20,   
                              contour_color="white",  
                              contour_width=2        
                             ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')  
        st.pyplot(fig)
    else:
        st.warning("No relevant emotional words found in the input texts.")

def main():
    st.set_page_config(page_title="Text Emotion Detection", page_icon="😊")

    
    st.title("Text Emotion Detection - NLP")
    st.subheader("Harish R - 21MIS1014 | Kishore Bharathi B - 21MIS1072")
    st.image("D:/header_image.png")  

   
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = []

    st.sidebar.header("")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here", height=200)
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if raw_text.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
           
            st.session_state.user_inputs.append(raw_text)

            # Prepare to display predictions for all user inputs
            predictions = []
            probabilities = []

            for text in st.session_state.user_inputs:
                prediction = predict_emotions(text)
                probability = get_prediction_proba(text)

                predictions.append(prediction)
                probabilities.append(probability)

            # Display results for all inputs
            st.success("All User Inputs and Predictions")
            result_df = pd.DataFrame({
                'Input Text': st.session_state.user_inputs,
                'Predicted Emotion': predictions,
                'Confidence (%)': [np.max(prob) * 100 for prob in probabilities]
            })

            st.dataframe(result_df)

            st.success("Prediction Probability for All Inputs")
            for idx, input_text in enumerate(st.session_state.user_inputs):
                proba_df = pd.DataFrame(probabilities[idx], columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotions", "Probability"]

                st.subheader(f"Probability Distribution for Input {idx + 1}")
                bar_chart = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='Emotions',
                    y='Probability',
                    color='Emotions'
                )
                st.altair_chart(bar_chart, use_container_width=True)

    
                pie_chart = alt.Chart(proba_df_clean).mark_arc().encode(
                    theta='Probability:Q',
                    color='Emotions:N'
                ).properties(width=400, height=400)
                st.altair_chart(pie_chart, use_container_width=True)

            st.subheader("WordCloud from All User Inputs")
            plot_custom_wordcloud(st.session_state.user_inputs)

if __name__ == "__main__":
    main()

