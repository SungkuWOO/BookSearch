import json

import requests
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

st.set_page_config(
    page_title="Woo's book",
    page_icon="📖",
)


pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index('books')
openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def get_embedding(text_list):
    response = openai_client.embeddings.create(
        input=text_list,
        model="text-embedding-3-small",
        dimensions=512
    )
    return [x.embedding for x in response.data]


def get_translation(query):
    url = "https://asia-northeast3-skilled-chalice-402604.cloudfunctions.net/translate"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"queries": [query]})
    translations = requests.post(
        url=url,
        data=payload,
        headers=headers
    )
    result = translations.json()
    translations = result["translations"]
    return translations[0]


def recommend(query_embedding):
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
    )
    matches = results["matches"]
    return [x["metadata"] for x in matches]


def generate_prompt(query, items):
    prompt = f"""
유저가 읽고 싶은 책에 대한 묘사와 이에 대한 추천 결과가 주어집니다.
유저의 입력과 각 추천 결과 책의 제목, 저자, 소개 등을 참고하여 추천사를 작성하세요.
당신에 대한 소개를 먼저 하고, 친절한 말투로 작성해주세요.
중간 중간 이모지를 적절히 사용해주세요.

---
유저 입력: {query}

추천 결과 1
제목: {items[0]['title']}
저자: {items[0]['authors']}
책소개: {items[0]['summary']}

추천 결과 2
제목: {items[1]['title']}
저자: {items[1]['authors']}
책소개: {items[1]['summary']}

추천 결과 3
제목: {items[2]['title']}
저자: {items[2]['authors']}
책소개: {items[2]['summary']}
---
"""
    return prompt


def request_chat_completion(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "당신은 책을 추천해주는 책방지기, Woo's 입니다."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    return response


def get_author_title(item):
    author = item["authors"]
    title = item["title"]
    author_list = author.split(",")
    if len(author_list) > 1:
        author = f"{author_list[0]} 외 {len(author_list) - 1}인"
    return f"{author} - {title}"


def process_recommend_results(items):
    st.markdown("**추천결과 🎁 (열 수 있어요!)**")
    for i, item in enumerate(items):
        with st.expander(f"#{i+1} {get_author_title(item)}"):
            st.header(item["title"])
            st.write(f"**{item['authors']}** | {item['publisher']} | {item['published_at']} | [yes24]({item['url']})")
            col1, col2 = st.columns([0.25, 0.75], gap="medium")
            with col1:
                st.image(item["img_url"])
            with col2:
                st.write(item["summary"])


def process_generated_text(streaming_resp):
    st.markdown("** Woo's 추천사 ✍️**")
    report = ""
    res_box = st.empty()
    for chunk in streaming_resp:
        delta = chunk.choices[0].delta
        if delta.content:
            report += delta.content
            res_box.markdown("".join(report).strip())
    return report


st.title("Woo's 책 안내 📖🐛")
#st.image("./images/banner.png")
st.image("banner.png")
with st.form("form"):
    query = st.text_input(
        label="읽고 싶은 책을 알려주면 AI가 추천해 줍니다요💡",
        placeholder="ex) 사랑과 미움을 다룬 이야기"
    )
    submitted = st.form_submit_button("제출")
if submitted:
    if not query:
        st.error("읽고 싶은 책 묘사를 작성해 주삼")
    else:
        with st.spinner("Woo's가 책을 찾고 있구만..."):
            translated_query = get_translation(query)
            query_embedding = get_embedding(translated_query)
            items = recommend(query_embedding)
        process_recommend_results(items)

        with st.spinner("Woo's가 추천사를 작성하는 중..."):
            prompt = generate_prompt(query, items)
            streaming_resp = request_chat_completion(prompt)
        process_generated_text(streaming_resp)
