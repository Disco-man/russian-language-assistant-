from dotenv import load_dotenv
import streamlit as st
import spacy
import pandas as pd
from PIL import Image
import pytesseract
import requests
import os
import base64
import io

# ====== spaCy ======
nlp = spacy.load("ru_core_news_sm")
load_dotenv("capi.env")
load_dotenv("gapi.env")

# ====== Переводы ======
pos_map = {
    "NOUN": "Существительное",
    "VERB": "Глагол",
    "AUX": "Вспомогательный глагол",  # ← добавлено
    "ADJ": "Прилагательное",
    "ADV": "Наречие",
    "PRON": "Местоимение",
    "ADP": "Предлог",
    "CCONJ": "Союз (сочинительный)",
    "SCONJ": "Союз (подчинительный)",
    "DET": "Определитель",
    "PART": "Частица",
    "NUM": "Числительное",
    "INTJ": "Междометие",
    "PROPN": "Имя собственное",
    "PUNCT": "Пунктуация",
    "SYM": "Символ",
    "X": "Другое"
}

role_map = {
    "nsubj": "Подлежащее",
    "nsubj:pass": "Подлежащее",
    "root": "Сказуемое",
    "aux": "Сказуемое",  # ← добавлено
    "obj": "Дополнение",
    "iobj": "Дополнение",
    "obl": "Обстоятельство",
    "obl:agent": "Обстоятельство",
    "advmod": "Обстоятельство",
    "amod": "Определение",
    "det": "Определение",
    "appos": "Приложение",
    "acl": "Определение",
    "case": "Обстоятельство",
    "conj": "—",
    "cc": "—",
    "nmod": "Дополнение",
    "nummod:gov": "Обстоятельство",  # ← добавлено
    "punct": "—"
}

morph_key_human = {
    "Case": "Падеж",
    "Number": "Число",
    "Gender": "Род",
    "Person": "Лицо",
    "Tense": "Время",
    "Aspect": "Вид",
    "Mood": "Наклонение",
    "VerbForm": "Форма глагола",
    "Voice": "Залог",
    "Degree": "Степень",
    "Animacy": "Одушевлённость"
}

morph_value_map = {
    "Case": {"Nom": "Именительный", "Acc": "Винительный", "Gen": "Родительный", "Dat": "Дательный", "Loc": "Предложный", "Ins": "Творительный"},
    "Number": {"Sing": "Ед. число", "Plur": "Мн. число"},
    "Gender": {"Masc": "Мужской", "Fem": "Женский", "Neut": "Средний"},
    "Person": {"1": "1-е лицо", "2": "2-е лицо", "3": "3-е лицо", "First": "1-е лицо", "Second": "2-е лицо", "Third": "3-е лицо"},
    "Tense": {"Pres": "Настоящее", "Past": "Прошедшее", "Fut": "Будущее"},
    "Aspect": {"Imp": "Несовершенный", "Perf": "Совершенный"},
    "Mood": {"Ind": "Изъявительное", "Imp": "Повелительное"},
    "VerbForm": {"Fin": "Спряжённая", "Inf": "Инфинитив", "Part": "Причастие", "Ger": "Деепричастие"},
    "Voice": {"Act": "Действительный", "Pass": "Страдательный", "Mid": "Средний"},  # ← добавлено
    "Degree": {"Pos": "Положительная", "Cmp": "Сравнительная", "Sup": "Превосходная"},
    "Animacy": {"Anim": "Одушевлённое", "Inan": "Неодушевлённое"}
}


def translate_morph(morph):
    try:
        feat_dict = morph.to_dict()
    except Exception:
        feat_dict = {}

    s = str(morph).strip()
    if s:
        for pair in s.split("|"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                feat_dict[k] = v

    if not feat_dict:
        return "—"

    parts = []
    for k, v in feat_dict.items():
        kh = morph_key_human.get(k, k)
        values = v.split(",") if isinstance(v, str) else [v]
        translated_values = []
        for val in values:
            val = val.strip()
            translated = None
            lookup = morph_value_map.get(k, {})
            for cand in (val, val.capitalize(), val.upper(), val.lower()):
                if cand in lookup:
                    translated = lookup[cand]
                    break
            if translated is None:
                translated = val
            translated_values.append(translated)
        parts.append(f"{kh}: {', '.join(translated_values)}")
    return "; ".join(parts)


# ====== Streamlit UI ======
def local_css():
    st.markdown(
        """
        <style>
        :root{
            --bg: #f4f8fb;
            --card: #ffffff;
            --accent: #3b5998;
            --accent-2: #60b6f0;
            --muted: #64748b;
            --radius: 14px;
        }
        .stApp { background: var(--bg); }
        .app-header {
            display: flex;
            align-items: center;
            gap: 16px;
            padding: 22px 18px;
            border-radius: var(--radius);
            background: linear-gradient(90deg, var(--accent), #60b6f0 80%);
            color: white;
            margin-bottom: 22px;
            box-shadow: 0 6px 18px rgba(30,58,138,0.08);
        }
        .app-logo {
            width: 54px;
            height: 54px;
            border-radius: 12px;
            background: var(--card);
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            font-size: 2rem;
            color: var(--accent);
            border: 2px solid var(--accent-2);
            box-shadow: 0 2px 8px #60b6f033;
        }
        .app-title { font-size: 26px; margin: 0; font-weight: 700;}
        .app-sub { color: #e0e7ef; margin: 0; font-size: 14px; opacity: 0.97 }
        .card {
            background: var(--card);
            border-radius: var(--radius);
            padding: 16px;
            box-shadow: 0 6px 18px rgba(30,58,138,0.04);
            margin-bottom: 14px;
        }
        /* --- КНОПКИ --- */
        .stButton>button {
            border-radius: 10px;
            padding: 10px 18px;
            font-weight: 600;
            background: #fff;
            color: var(--accent);
            border: 1.5px solid var(--accent-2);
            transition: background 0.2s, color 0.2s;
        }
        .stButton>button:hover {
            background: var(--accent-2);
            color: #fff;
        }
        .muted { color: var(--muted); font-size: 13px; }
        .ocr-output {
            background: var(--accent);
            color: #f8fafc;
            padding: 14px;
            border-radius: 12px;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .custom-label {
            font-size: 20px;
            font-weight: 600;
            margin-bottom: 2px !important;
            margin-top: 0 !important;
            line-height: 1.2;
        }
        .custom-label { margin-bottom: 0px !important; }
        @media (max-width: 720px) {
            .app-header { flex-direction: column; align-items: flex-start; gap: 8px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

st.set_page_config(page_title="Помощник по русскому", layout="wide")

st.markdown(
    """
    <div class="app-header">
        <div class="app-logo">РУ</div>
        <div>
            <div class="app-title">Помощник по русскому языку</div>
            <div class="app-sub">HTR • Морфология • Синтаксис • Генерация текстов • Проверка анализа</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# Увеличиваем размер надписей для загрузки файла и текстового поля
st.markdown('<div class="custom-label">📂Перетащите фото с текстом сюда или нажмите, чтобы выбрать</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if "ocr_text" not in st.session_state:
    st.session_state["ocr_text"] = ""

def gpt_vision_ocr(image: Image.Image, api_key: str) -> str:
    import io, base64, requests

    # Сохраняем изображение в память
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": "gpt-4o-mini",  # Можно также gpt-4o (основная), но mini быстрее и дешевле
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Извлеки весь читаемый текст с этого изображения. Не добавляй ничего лишнего, только текст."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
                ]
            }
        ],
        "max_output_tokens": 1024
    }

    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=data,
        timeout=60
    )

    if response.status_code == 200:
        result = response.json()
        return result["output"][0]["content"][0]["text"].strip()
    else:
        return f"Ошибка Vision API: {response.status_code}\n{response.text}"


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    if st.button("📖 Распознать текст"):
        API_KEY = os.getenv("OPENAI_API_KEY")
        if not API_KEY:
            st.error("❌ Нет API ключа для нейросети.")
        else:
            with st.spinner("Помощник распознаёт текст..."):
                ocr_text = gpt_vision_ocr(image, API_KEY)
                st.session_state["ocr_text"] = ocr_text
                st.success("Текст распознан!")
                st.write(ocr_text)

default_text = st.session_state["ocr_text"] if st.session_state["ocr_text"] else ""
st.markdown('<div class="custom-label">Введите текст для анализа:</div>', unsafe_allow_html=True)
text = st.text_area("", value=default_text, height=150)

# ====== Морфологический и синтаксический разбор ======
if st.button("🔎 Разобрать предложение"):
    doc = nlp(text)
    rows = []
    for token in doc:
        if token.is_space or token.is_punct:
            continue
        pos = pos_map.get(token.pos_, token.pos_)
        role = role_map.get(token.dep_.lower(), token.dep_)
        rows.append({
            "Словоㅤㅤ": token.text,
            "Леммаㅤㅤ": token.lemma_,
            "Часть речи": pos,
            "Роль в предложении": role,
            "Морф. признаки": translate_morph(token.morph)
        })
    if rows:
        st.subheader("Синтаксический и морфологический разбор")
        st.table(pd.DataFrame(rows))

# ====== Глубокий анализ текста (GPT / API) ======
API_KEY = os.getenv("OPENAI_API_KEY")  # вставь свой ключ в переменную окружения


if st.button("🧠 Анализ текста для экзамена"):
    import google.generativeai as genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("❌ Нет API ключа Gemini.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Ты — эксперт по русскому языку, готовишь учеников к экзаменам. "
            "Отвечай строго по школьным критериям. Не выдумывай, если не уверен. "
            "Пример хорошего ответа:\n"
            "Тема: Тема — это то, о чём идёт речь, в 1 предложении. Тема не должна быть длинной. Пример: Тема текста - сохранение природы и ответственность человека за окружающую среду.\n"
            "Идея: Это главная мысль автора, то, зачем он написал текст — **не то, что он делает**, а **что он хочет донести до читателя как вывод или мораль**. Формулируй идею как обобщённую мысль, выражающую позицию автора, часто в форме утверждения или философского вывода. Идея должна быть не слишком длинной, без лишних деталей. Пример: только осознанное отношение человека к природе способно предотвратить экологическую катастрофу.\n"
            "Цель текста: Это зачем создан текст — информировать, убедить, рассказать, побудить к действию, вызвать эмоцию и т.д. Пример: 'Цель текста — убедить читателя в необходимости заботы о природе.'\n"
            "Жанр: «Жанр — это форма текста (например: статья, сочинение-рассуждение, эссе, доклад и т.п.), \n"
            "Стиль: Определи, в каком стиле написан текст: художественный, публицистический, научный, официально-деловой, разговорный.\n"
            "Признаки стиля: «Признаки стиля укажи с примерами слов или выражений из текста. Не пиши общие слова без примеров.»\n"
            "Языковые особенности: Языковые особенности проанализируй на основе конкретных примеров из текста. "
            "Не пиши одно и то же в разных пунктах. Роль проанализируй конкретно и подробно. При анализе лексики старайся выбирать первым примером вид лексики(Лексика делится на общеупотребительную, просторечную, диалектную, профессиональную, терминологическую, разговорную, книжную, поэтическую, архаическую, неологизмы, заимствованную и жаргонную.), вторым примером изобразительно-выразительные средства(не делай этого если изобразительно-выразительные средства отсутствуют)\n\n"
            f"Теперь проанализируй следующий текст:\n{text}\n"
            "Сначала представь в структурированном виде тему, идею, цель, целевую аудиторию, жанр, стиль, признаки стиля с примерами из текста и объяснением их роли, "
            "а также языковые особенности: два примера лексики (лексика включает виды лексики и изобразительно-выразительные средства) с указанием их роли, "
            "два примера синтаксиса с объяснением их роли и два примера морфологии с пояснением их роли. "
            "Затем составь итоговый развернутый анализ в виде связного школьного текста, как это пишут на экзамене. "
            "Итоговый анализ должен включать все вышеуказанные пункты, быть написан в публицистически-академическом стиле, легко читаемом для школьника, без излишней сухости и академичности."
        )
        with st.spinner("Помощник анализирует текст..."):
            try:
                response = model.generate_content(prompt)
                result = response.text
                st.subheader("Результаты анализа")
                st.write(result)
            except Exception as e:
                st.error(f"Ошибка Gemini: {e}")

# ====== Генерация текста для практики ======
if "practice_text" not in st.session_state:
    st.session_state["practice_text"] = ""

if st.button("🎲 Сгенерировать текст для анализа"):
    import google.generativeai as genai

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("❌ Нет API ключа Gemini.")
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "Сгенерируй связный текст на русском языке для анализа. "
            "Выбери стиль случайно из трёх: публицистический, научный или официально-деловой (не указывай стиль явно в тексте). "
            "Тема — любая, текст должен быть реалистичным, подходить для экзаменационного анализа, объёмом примерно 250 слов. "
            "Не пиши отдельно тему или стиль, просто выдай сам текст."
        )
        with st.spinner("Помощник генерирует текст..."):
            try:
                response = model.generate_content(prompt)
                st.session_state["practice_text"] = response.text
            except Exception as e:
                st.error(f"Ошибка Gemini: {e}")

if st.session_state["practice_text"]:
    st.subheader("📖 Текст для практики анализа")
    st.write(st.session_state["practice_text"])

    user_analysis = st.text_area("✍ Напиши свой анализ этого текста:", height=200)

    if st.button("✅ Проверить мой анализ"):
        import google.generativeai as genai

        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            st.error("❌ Нет API ключа Gemini.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = (
                "Ты — эксперт по русскому языку, проверяешь школьные анализы текстов. "
                "Отвечай чётко, по пунктам, в стиле школьного учителя.\n\n"
                f"Вот текст для анализа:\n{st.session_state['practice_text']}\n\n"
                f"Вот ученический анализ:\n{user_analysis}\n\n"
                "Проверь этот анализ: отметь, что сделано правильно, чего не хватает (тема, идея, жанр, стиль, признаки стиля с примерами, языковые особенности — 2 лексики, 2 синтаксиса, 2 морфологии), "
                "и дай советы, как улучшить ответ. Стиль ответа — как у школьного учителя: структурированный и понятный."
            )
            with st.spinner("Помощник проверяет анализ..."):
                try:
                    response = model.generate_content(prompt)
                    feedback = response.text
                    st.subheader("🔍 Проверка твоего анализа")
                    st.write(feedback)
                except Exception as e:
                    
                    st.error(f"Ошибка Gemini: {e}")

